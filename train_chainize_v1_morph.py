import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np

from copy import deepcopy

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse
import time
import sys

from models import model_pool
from models.util import prepare_model

from data.dataset_adience_vanilla import AdienceVanilla
from data.dataset_adience_validation_partial_node import AdienceReference
from data.dataset_chainization_adience import ChainizedAdience
from data.dataset_adience_partial_order_baselineddddd import AdiencePartial
from util import adjust_learning_rate, accuracy, cls_accuracy_bc, AverageMeter, ClassWiseAverageMeter, cross_entropy_loss_with_one_hot_labels, mix_ce_and_kl_loss
from util import write_log, get_current_time, to_np

from visualize_utils import label2color_10
from chainize_utils import compute_chainize_error, generate_training_pairs_from_chain, generate_training_pairs_by_transitivity, extract_features, chainize_by_modified_Kahn_sort_quick2, refine_chain, analyze_generated_pairs
import wandb
from scipy.stats import spearmanr, kendalltau


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--info_ratio', type=int, default=0.0015, help='information ratio')
    parser.add_argument('--use_transitivity', type=bool, default=False, help='whether to use transitivity trick')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,100,150', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', type=bool, default=True, help='use adam optimizer')

    # network
    parser.add_argument('--backbone', type=str, default='vgg16', choices=model_pool)
    parser.add_argument('--model', type=str, default='BinaryV5')

    # parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--train_file', type=str, default='/hdd/2021/Research/4_OL/partially_ordered_data_OL/data/MORPH/SettingA_F%d_train_[16_77]_256size.pickle', help='path to train file')
    parser.add_argument('--test_file', type=str, default='/hdd/2021/Research/4_OL/partially_ordered_data_OL/data/MORPH/SettingA_F%d_test_[16_60]_256size.pickle', help='path to test file')

    parser.add_argument('-t', '--trial', type=str, default='chainize', help='the experiment id')
    parser.add_argument('--tags', type=str, default="Adience, chainize", help='add tags for the experiment')

    opt = parser.parse_args()

    # set the path according to the environment

    opt.train_file = opt.train_file % (opt.fold)
    opt.test_file = opt.test_file % (opt.fold)

    if not opt.model_path:
        opt.model_path = './models_trained/Adience/Rebuttal_timing'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = '{}_{}_lr_{}_F{}_IR{}_{}'.format(opt.backbone, opt.model, opt.learning_rate, opt.fold, opt.info_ratio, get_current_time())

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    log_file = open(f'{opt.save_folder}/log.txt', 'w')
    opt_dict = vars(opt)
    for key in opt_dict.keys():
        write_log(log_file, f'{key}: {opt_dict[key]}')

    opt.n_gpu = torch.cuda.device_count()
    return opt, log_file


def main():
    # to reproduce
    np.random.seed(999)

    opt, log_file = parse_option()
    # wandb.login(key='1ba025567068b512f5d1a4125a11e7e7cb62fe9c')
    # wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
    # wandb.config.update(opt)
    # wandb.save('*.py')
    # wandb.run.save()

    # dataloader
    train_data = AdiencePartial(opt.train_file,  norm_age=True, info_ratio=opt.info_ratio, build_graph=True)

    test_loader_ref_train = DataLoader(AdienceReference(opt.test_file, opt.train_file, use_age=True),
                                       batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                       num_workers=opt.num_workers)

    vanilla_train_loader = DataLoader(AdienceVanilla(opt.train_file),
                                      batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                      num_workers=opt.num_workers)

    chainize_results = []
    #Graph info
    print(f'Number of nodes : {len(train_data.G.nodes)}')
    chainized_order = nx.topological_sort(train_data.G)
    print(f'==> Kahn algorithm result!!')
    chainized_ages = [train_data.G.nodes[node]["ages"][0] for node in chainized_order]
    err = compute_chainize_error(chainized_ages)
    write_log(log_file, f'\nKahn algorithm chainize err : {err:.4f}\n')
    chainize_results.append(chainized_ages)

    print(chainized_ages)
    gt = deepcopy(chainized_ages)
    gt.sort()

    rho = spearmanr(chainized_ages, gt)
    k_tau = kendalltau(chainized_ages, gt)
    write_log(log_file, f'kahn : spearmanr : {rho[0]:.5f}   // k-tau : {k_tau[0]:.5f}')

    np.save(f'{opt.save_folder}/kahn_e{err:.4f}.npy', chainized_ages)

    if opt.use_transitivity:
        pair_set = generate_training_pairs_by_transitivity(train_data.G)
        train_data.add_generated_pairs(pair_set)

    # model
    model = prepare_model(opt)
    # wandb.watch(model)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = cross_entropy_loss_with_one_hot_labels

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

    # routine: weakly supervised training
    best = 80
    val_best = 60
    log_dict = dict()
    chainize_flag1 = True
    chainize_flag2 = False

    train_data.epoch = -1
    for epoch in range(opt.epochs):
        if chainize_flag1:
            train_data.epoch += 1
            print(f"==> update train loader... epoch {train_data.epoch}")
            train_loader = DataLoader(train_data,
                                      batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                      num_workers=opt.num_workers)
        else:
            pseudo_data.epoch += 1
            pseudo_data.epoch_gen += 1
            if pseudo_data.epoch_gen > 20:
                chainize_flag2 = True
            print("==> update train loader using pseudo data... ")
            train_loader = DataLoader(pseudo_data,
                                      batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                      num_workers=opt.num_workers)

        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.val_freq == 0:
            print('==> Test using training data as reference')
            test_acc = validate(test_loader_ref_train, model, opt)
            # print('==> Test using test data as reference')
            # test_acc2 = validate(phase, test_loader_ref_test, model, opt)
            if test_acc > val_best:
                val_best = test_acc
                # save the last model
                state = {
                    'opt': opt,
                    'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                    'pseudo_ranks': train_data.labels
                }
                save_file = os.path.join(opt.save_folder,
                                         f'epoch_{epoch}_val_best_{val_best:.3f}.pth')
                torch.save(state, save_file)
                print(
                    f'*** models have been saved as epoch_{epoch}_val_best_{val_best:.3f}.pth')

        if train_acc > best:
            best = train_acc
            # save the last model
            state = {
                'opt': opt,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'pseudo_ranks': train_data.labels
            }
            save_file = os.path.join(opt.save_folder, f'epoch_{epoch}_train_best_{best:.3f}.pth')
            torch.save(state, save_file)
            print(f'*** models at have been saved as epoch_{epoch}_train_best_{best:.3f}.pth.')

        # chainize
        if train_acc > 90 and (chainize_flag1 or chainize_flag2):
            chainize_flag1 = False
            chainize_flag2 = False
            t1 = time.time()
            embs = extract_features(model.encoder, vanilla_train_loader)
            t2 = time.time()
            chainized_order = chainize_by_modified_Kahn_sort_quick2(deepcopy(train_data.G), model.comparator, embs)
            t3 = time.time()
            print(f'==> Epoch {epoch} chainization result!!')
            chainized_ages = [train_data.G.nodes[node]["ages"][0] for node in chainized_order]
            chainize_results.append(chainized_ages)
            err = compute_chainize_error(chainized_ages)
            write_log(log_file, f'chainize err : {err:.4f}')
            print(chainized_ages)

            # plt.figure(figsize=(20, 20))
            # for i in range(len(chainize_results)):
            #     if i == 0:
            #         plt.plot(chainize_results[i], color=label2color_10(i), label='Kahn')
            #     else:
            #         plt.plot(chainize_results[i], color=label2color_10(i), label=f'MD-Kahn_ep{(i-1)*20}')
            # plt.plot(gt, color=label2color_10(9), label='GT')
            # plt.grid()
            # plt.legend()
            # plt.savefig(f'{opt.save_folder}/IR{opt.info_ratio}_chainize_comparison_ep{epoch}.png')
            # plt.close()
            #
            # plt.figure(figsize=(20, 20))
            # plt.plot(chainized_ages, color=label2color_10(len(chainize_results)), label=f'MD-Kahn_ep{(len(chainize_results) - 1) * 20}')
            # plt.plot(gt, color=label2color_10(9), label='GT')
            # plt.grid()
            # plt.legend()
            # plt.title(f'err: {err:.4f}')
            # plt.savefig(f'{opt.save_folder}/IR{opt.info_ratio}_chainize_result_ep{epoch}.png')
            # plt.close()

            rho = spearmanr(chainized_ages, gt)
            k_tau = kendalltau(chainized_ages, gt)
            write_log(log_file, f'spearmanr : {rho[0]:.5f}   // k-tau : {k_tau[0]:.5f}')

            np.save(f'{opt.save_folder}/ep{epoch}_e{err:.4f}.npy', chainized_ages)

            print('==> chain shortening start')
            chain_info = deepcopy([train_data.G.nodes[node]["img_inds"] for node in chainized_order])

            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'pseudo_ranks': train_data.labels,
                'chain_info':chain_info
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}_chainize.pth')
            torch.save(state, save_file)

            # merge nodes!!
            t4 = time.time()
            chain_info, age_diff, edge_val = refine_chain(model.comparator, embs, chain_info,
                                                          set(train_data.pair_list), node_threshold=8 * 100,
                                                          age_data=train_data.ages)
            t5 = time.time()
            write_log(log_file, f'shortend chain length: {len(chain_info)}')
            write_log(log_file, f'mean_age_diff: {np.mean(age_diff)}')

            write_log(log_file, f'feature extraction : {t2-t1:.3f}')
            write_log(log_file, f'sorting : {t3 - t2:.3f}')
            write_log(log_file, f'merging : {t5 - t4:.3f}')

            print('==> generate training pairs based on chainization')
            generated_pair_list, pseudo_labels = generate_training_pairs_from_chain(chain_info,
                                                                                    set(train_data.pair_list),
                                                                                    train_data.n_imgs,
                                                                                    min_distance=int(np.ceil(len(chain_info)/8)*1.7))
            write_log(log_file, f'Epoch[{epoch}], generated pairs: {len(generated_pair_list)}')
            # analysis


            analyze_generated_pairs(log_file, generated_pair_list, pseudo_labels, train_data.ages)
            # np.save(f'age_diff_ep{epoch}.npy', age_diff)
            # np.save(f'ox_list_ep{epoch}.npy', ox_list)

            pseudo_data = ChainizedAdience(epoch, train_data.imgs, train_data.ages, train_data.labels)
            pseudo_data.update_pairs(annotated_pair_list=train_data.pair_list,
                                     order_labels=train_data.order_list,
                                     generated_pair_list=generated_pair_list,
                                     pseudo_labels=pseudo_labels)


        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'pseudo_ranks': train_data.labels
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, save_file)

        log_dict['epoch'] = epoch
        log_dict['Train Acc'] = train_acc
        log_dict['Train Loss'] = train_loss
        log_dict['Test Acc'] = test_acc
        # wandb.log(log_dict)

    # remove output.txt log file
    output_log_file = os.path.join(wandb.run.dir, "output.log")
    if os.path.isfile(output_log_file):
        os.remove(output_log_file)
    else:  ## Show an error ##
        print("Error: %s file not found" % output_log_file)

    print('done')


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = ClassWiseAverageMeter(3)

    end = time.time()
    for idx, (x_base, x_ref, one_hot_labels, order_labels, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            x_base = x_base.cuda()
            x_ref = x_ref.cuda()
            one_hot_labels = one_hot_labels.cuda()
            order_labels = order_labels.cuda()
            # batch_size = x_base.size(0)
            # mask = np.ones([batch_size, ], dtype=bool)
            # mask[np.argwhere(to_np(order_labels) == -1).flatten()] = False

        # ===================forward=====================
        logits = model(x_base, x_ref)
        total_loss = criterion(logits, one_hot_labels)#, mask, alpha=3)

        # losses.append(phase_i_loss)
        acc, cnt = cls_accuracy_bc(nn.functional.softmax(logits, dim=-1), order_labels)
        top1.update(acc, cnt)
        losses.update(total_loss.item(), x_base.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print(
                  f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f}\t'
                  f'Data {data_time.val:3f}\t'
                  f'Loss {losses.val:.4f}\t'
                  f'Acc [{top1.val[0]:.3f}  {top1.val[1]:.3f}  {top1.val[2]:.3f}]  [{top1.total_avg:.3f}]'
                  )
            sys.stdout.flush()

    print(f' * Acc@1 [{top1.avg[0]:.3f}  {top1.avg[1]:.3f}   {top1.avg[2]:.3f}]  [{top1.total_avg:.3f}]\n')

    return top1.total_avg, losses.avg


def validate(test_loader, model, opt):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = ClassWiseAverageMeter(3)

    with torch.no_grad():
        end = time.time()

        for idx, (x_base, x_ref, order_labels, _, _) in enumerate(test_loader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                x_base = x_base.cuda()
                x_ref = x_ref.cuda()
                order_labels = order_labels.cuda()

            # ===================forward=====================
            logits = model(x_base, x_ref)

            # losses.append(phase_i_loss)
            acc, cnt = cls_accuracy_bc(nn.functional.softmax(logits, dim=-1), order_labels)
            top1.update(acc, cnt)

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % opt.print_freq == 0:
                print(f'==> TEST [{idx}]/[{len(test_loader)}]')

        print(f' * Test Acc@1 [{top1.avg[0]:.3f}  {top1.avg[1]:.3f}   {top1.avg[2]:.3f}]  [{top1.total_avg:.3f}]\n')
        sys.stdout.flush()
    return top1.total_avg


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

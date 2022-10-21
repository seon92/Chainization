from .resnet import resnet12, resnet18
import torchvision.models as models

model_pool = [
    'resnet12',
    'resnet18',
    'vgg16',
]

model_dict = {
    'resnet12': resnet12,
    'resnet18': resnet18,
    'vgg16':models.vgg16_bn(pretrained=True)
}

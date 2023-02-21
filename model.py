""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(backbone, outputchannels=1):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    if backbone == 'mobilenetv3':
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
    elif backbone == 'resnet50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    elif backbone == 'resnet101':
            model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)    
#     model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
#     model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

import torch.nn as nn
from torchvision import models

def print_layers(model):
    children_counter = 0
    for n,c in model.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1
    return

def load_model(backbone, weights="None"):
    if backbone == "alexnet":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.alexnet(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.alexnet(pretrained=True)
        else:
            model = models.alexnet()
    elif backbone == "densenet121":
        output_layer = "features"
        if weights == "imagenet":
            model = models.densenet121(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121()
    elif backbone == "densenet161":
        output_layer = "features"
        if weights == "imagenet":
            model = models.densenet161(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.densenet161(pretrained=True)
        else:
            model = models.densenet161()
    elif backbone == "densenet169":
        output_layer = "features"
        if weights == "imagenet":
            model = models.densenet169(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.densenet169(pretrained=True)
        else:
            model = models.densenet169()
    elif backbone == "densenet201":
        output_layer = "features"
        if weights == "imagenet":
            model = models.densenet201(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.densenet201(pretrained=True)
        else:
            model = models.densenet201()
    elif backbone == "efficientnet_v2_l":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.efficientnet_v2_l(weights="DEFAULT")
        elif weights == "legacy":
            model = models.efficientnet_v2_l(pretrained=True)
        else:
            model = models.efficientnet_v2_l()
    elif backbone == "efficientnet_v2_m":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.efficientnet_v2_m(weights="DEFAULT")
        elif weights == "legacy":
            model = models.efficientnet_v2_m(pretrained=True)
        else:
            model = models.efficientnet_v2_m()
    elif backbone == "efficientnet_v2_s":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.efficientnet_v2_s(weights="DEFAULT")
        elif weights == "legacy":
            model = models.alexnet(pretrained=True)
        else:
            model = models.efficientnet_v2_s()
    elif backbone == "inception_v3":
        output_layer = "dropout"
        if weights == "imagenet":
            model = models.inception_v3(weights="DEFAULT")
        elif weights == "legacy":
            model = models.inception_v3(pretrained=True)
        else:
            model = models.inception_v3()
    elif backbone == "maxvit_t":
        output_layer = "blocks"
        if weights == "imagenet":
            model = models.maxvit_t(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.maxvit_t(pretrained=True)
        else:
            model = models.maxvit_t()
    elif backbone == "mobilenet_v2":
        output_layer = "features"
        if weights == "imagenet":
            model = models.mobilenet_v2(weights="IMAGENET1K_V2")
        elif weights == "legacy":
            model = models.mobilenet_v2(pretrained=True)
        else:
            model = models.mobilenet_v2()
    elif backbone == "mobilenet_v3_small":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.mobilenet_v3_small(pretrained=True)
        else:
            model = models.mobilenet_v3_small()
    elif backbone == "mobilenet_v3_large":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        elif weights == "legacy":
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
    elif backbone == "resnet18":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
    elif backbone == "resnet34":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.resnet34(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.resnet34(pretrained=True)
        else:
            model = models.resnet34()
    elif backbone == "resnet50":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.resnet50(weights="IMAGENET1K_V2")
        elif weights == "legacy":
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
    elif backbone == "resnet101":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.resnet101(weights="IMAGENET1K_V2")
        elif weights == "legacy":
            model = models.resnet101(pretrained=True)
        else:
            model = models.resnet101()
    elif backbone == "resnet152":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.resnet152(weights="IMAGENET1K_V2")
        elif weights == "legacy":
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
    elif backbone == "swin_v2_t":
        output_layer = "flatten"
        if weights == "imagenet":
            model = models.swin_v2_t(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.swin_v2_t(pretrained=True)
        else:
            model = models.swin_v2_t()
    elif backbone == "swin_v2_s":
        output_layer = "flatten"
        if weights == "imagenet":
            model = models.swin_v2_s(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.swin_v2_s(pretrained=True)
        else:
            model = models.swin_v2_s()
    elif backbone == "swin_v2_b":
        output_layer = "flatten"
        if weights == "imagenet":
            model = models.swin_v2_b(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.swin_v2_b(pretrained=True)
        else:
            model = models.swin_v2_b()
    elif backbone == "vgg16":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.vgg16(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.vgg16(pretrained=True)
        else:
            model = models.vgg16()
    elif backbone == "vgg19":
        output_layer = "avgpool"
        if weights == "imagenet":
            model = models.vgg19(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.vgg19(pretrained=True)
        else:
            model = models.vgg19()
    elif backbone == "vit_b_16":
        output_layer = "encoder"
        if weights == "imagenet":
            model = models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
        elif weights == "legacy":
            model = models.vit_b_16(pretrained=True)
        else:
            model = models.vit_b_16()
    elif backbone == "vit_b_32":
        output_layer = "encoder"
        if weights == "imagenet":
            model = models.vit_b_32(weights="IMAGENET1K_V1")
        elif weights == "legacy":
            model = models.vit_b_32(pretrained=True)
        else:
            model = models.vit_b_32()

    return model, output_layer

class siamese_embeddings(nn.Module):
    def __init__(self, backbone="vgg16", weights="None"):
        super().__init__()
        # Get inputs
        self.backbone = backbone
        self.weights = weights

        # Load pre-trained model
        model, output_layer = load_model(self.backbone, self.weights)

        # Remove Fully Connected Layers
        layers = list(model._modules.keys())
        layer_count = layers.index(output_layer) + 1
        for layer in layers[layer_count:]:
            dummy_var = model._modules.pop(layer)

        # Set Network
        self.net = nn.Sequential(model._modules)
        self.norm = nn.functional.normalize

    def forward(self,x):
        bs = x.shape[0]
        x = self.net(x).view(bs,-1)
        x = self.norm(x, p=2, dim=1)
        return x

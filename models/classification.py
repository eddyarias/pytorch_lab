import torch.nn as nn
from models.siamese import load_model

class classification_model(nn.Module):
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

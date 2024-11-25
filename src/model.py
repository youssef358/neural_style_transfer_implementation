import torch
import torchvision.models as models

class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_map_indices = {
            'conv1_1': 0,
            'conv2_1': 5,
            'conv3_1': 10,
            'conv4_1': 19,
            'conv4_2': 21,
            'conv5_1': 28,
        }


        self.slices = []
        start = 0
        for layer_idx in self.feature_map_indices.values():
            self.slices.append(torch.nn.Sequential(self.vgg_pretrained_features[start:layer_idx + 1]))
            start = layer_idx + 1
        self.slices = torch.nn.ModuleList(self.slices)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = {}
        for name, slice_layer in zip(self.feature_map_indices.keys(), self.slices):
            x = slice_layer(x)
            outputs[name] = x

        return outputs


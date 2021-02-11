import torch
import torch.nn as nn
import torchvision
import geffnet


class MobileNetV2Encoder(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = []
        
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)

        self.layers.append(nn.Sequential())
        
        layer = [model.features[0],
                model.features[1]]
        for index in range(2, len(model.features) - 1):
            if model.features[index].conv[1][0].stride == (2, 2):
                self.layers.append(nn.Sequential(*layer))
                layer = []
            layer.append(model.features[index])
        self.layers.append(nn.Sequential(*layer))

        self.eval()
        x = torch.zeros(1, 3, 224, 224)

        for layer in self.layers:
            x = layer(x)
            self.out_channels.append(x.shape[1])

    def forward(self, input):
        x = input
        features = []

        for index in range(len(self.layers)):
            x = self.layers[index](x)
            features.append(x)

        return features
    

class EfficientNetLite0Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = geffnet.tf_efficientnet_lite0(pretrained=True)
        self.out_channels = [3, 16, 24, 40, 80, 320]
        
    def forward(self, input):
        results = []
        
        res = input
        results.append(res)
        res = self.model.conv_stem(res)
        res = self.model.bn1(res)
        res = self.model.act1(res)

        res = self.model.blocks._modules['0'](res)
        results.append(res)

        res = self.model.blocks._modules['1'](res)
        results.append(res)

        res = self.model.blocks._modules['2'](res)
        results.append(res)

        res = self.model.blocks._modules['3'](res)
        results.append(res)

        res = self.model.blocks._modules['4'](res)
        res = self.model.blocks._modules['5'](res)
        res = self.model.blocks._modules['6'](res)
        results.append(res)

        return results

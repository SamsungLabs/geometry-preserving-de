import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_planes)
    )


class CRPBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(out_planes if (i == 0) else out_planes, out_planes, stride=1))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.adapt = conv1x1(in_planes, out_planes, 1)

    def forward(self, x):
        x = self.adapt(x)
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x / (self.n_stages + 1)


class RefineDecoder(torch.nn.Module):
    def __init__(self, backbone, block=(lambda x: CRPBlock(x, x, 2)), start_level=1, n_channels=1, branch_mode=False):
        super(RefineDecoder, self).__init__()
        self.backbone = backbone
        self.start_level = start_level
        self.n_channels = n_channels
        self.branch_mode = branch_mode

        self.up_branch = []
        self.adapters = []
        self.out_channels = list(backbone.out_channels)

        for index in range(start_level, len(backbone.out_channels)):
            out_features = backbone.out_channels[index]
            in_features = backbone.out_channels[index]
            
            self.out_channels[index] = out_features

            if index + 1 < len(backbone.out_channels):
                in_features = backbone.out_channels[index + 1]

                adapter = torch.nn.Sequential(
                    nn.Conv2d(in_features, out_features, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                self.adapters.append(adapter)
                
            up_branch = [block(out_features)]
                
            self.up_branch.append(nn.Sequential(*up_branch))
        
        self.up_branch = nn.ModuleList(self.up_branch[::-1])
        self.adapters = nn.ModuleList(self.adapters[::-1])
        self.final_decoder = [nn.Conv2d(backbone.out_channels[start_level], self.n_channels, 1)]

        for index in range(start_level):
            self.final_decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.final_decoder = nn.Sequential(*self.final_decoder)

    def forward(self, input):
        reversed_input = input[::-1]

        result = None
        results = []
        for index in range(len(self.up_branch)):
            if result is None:
                result = self.up_branch[index](reversed_input[index])
            else:
                result = self.up_branch[index](self.adapters[index - 1](result) + reversed_input[index])
                
            results.append(result)
        
        for index in reversed(range(self.start_level)):
            results.append(input[index])

        return self.final_decoder(result)

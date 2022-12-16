import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DnCNN(nn.Module):
    """
    Network architecture from this reference. 

    @article{zhang2017beyond,
      title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
      author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
      journal={IEEE Transactions on Image Processing},
      year={2017},
      volume={26},
      number={7},
      pages={3142-3155},
    }
    """

    def __init__(self, in_channels=3, out_channels=3, depth=17, hidden_channels=64,
                use_bn=True, use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=use_bias))
            if use_bn: layers.append(torch.nn.BatchNorm2d(hidden_channels))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = x
        residual = self.net(x)
        return y - residual


class ConvDAE(nn.Module):
    '''
    Network architecture from this reference. Adapted from https://github.com/yjn870/REDNet-pytorch

    @article{DBLP:journals/corr/MaoSY16a,
    author    = {Xiao{-}Jiao Mao and
                Chunhua Shen and
                Yu{-}Bin Yang},
    title     = {Image Restoration Using Convolutional Auto-encoders with Symmetric
                Skip Connections},
    journal   = {CoRR},
    volume    = {abs/1606.08921},
    year      = {2016},
    url       = {http://arxiv.org/abs/1606.08921},
    eprinttype = {arXiv},
    eprint    = {1606.08921},
    timestamp = {Mon, 13 Aug 2018 16:46:23 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/MaoSY16a.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    '''
    def __init__(self, in_channels=3, out_channels=3, num_layers=5, num_features=64, use_bias=True):
        super(ConvDAE, self).__init__()
        conv_layers = []
        deconv_layers = []

        # encoding layers
        conv_layers.append(nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                         nn.ReLU(inplace=True), 
                                         ))
                                         #nn.BatchNorm2d(num_features)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                             nn.ReLU(inplace=True), 
                                             ))
                                             #nn.BatchNorm2d(num_features)))

        # decoding layers
        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
                                               nn.ReLU(inplace=True), 
                                               ))
                                               #nn.BatchNorm2d(num_features)))

        deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        resid = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += resid
        out = self.relu(out)
        return out


# class REDNet20(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, num_layers=10, num_features=64, use_bias=True):
#         super(REDNet20, self).__init__()
#         self.num_layers = num_layers

#         conv_layers = []
#         deconv_layers = []

#         conv_layers.append(nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                                         nn.ReLU(inplace=True), 
#                                         #))
#                                         nn.BatchNorm2d(num_features)))

#         for i in range(num_layers - 1):
#             conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
#                                             nn.ReLU(inplace=True), 
#                                             #))
#                                             nn.BatchNorm2d(num_features)))

#         for i in range(num_layers - 1):
#             deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=use_bias),
#                                                 nn.ReLU(inplace=True), 
#                                                 #))
#                                                 nn.BatchNorm2d(num_features)))

#         deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias))

#         self.conv_layers = nn.Sequential(*conv_layers)
#         self.deconv_layers = nn.Sequential(*deconv_layers)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x

#         conv_feats = []
#         for i in range(self.num_layers):
#             x = self.conv_layers[i](x)
#             if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
#                 conv_feats.append(x)

#         conv_feats_idx = 0
#         for i in range(self.num_layers):
#             x = self.deconv_layers[i](x)
#             if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
#                 conv_feat = conv_feats[-(conv_feats_idx + 1)]
#                 conv_feats_idx += 1
#                 x = x + conv_feat
#                 x = self.relu(x)

#         x += residual
#         x = self.relu(x)

#         return x


# some reference
# class DAE(nn.Module):
#     def __init__(self, num_features=64):
#         super(DAE, self).__init__()
#         ## encoder layers ##
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         ## decoder layers ##
#         self.t_conv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
#         self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

#     def forward(self, x):
#         ## encode ##
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
        
#         ## decode ##
#         # add transpose conv layers, with relu activation function
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = torch.sigmoid(self.conv_out(x))
                
#         return x

from pyexpat import features

import _init_paths
import lib
import torch
import torch.nn as nn
class VOSNet(nn.Module):
    def __init__(self, opt):
        super(VOSNet, self).__init__()
        self.opt = opt
        self.bn = nn.BatchNorm2d
        self.num_points = opt.num_points

        if opt.encoder == 'swin_tiny': 
            self.backbone_x = lib.swin_tiny()
            self.backbone_y = lib.swin_tiny()
        elif opt.encoder == 'mit_b2':
            self.backbone_x = lib.mit_b2()
            self.backbone_y = lib.mit_b2()
        elif opt.encoder == 'efficientnetV2':
            self.backbone_x = lib.efficientnetv2_s()
            self.backbone_y = lib.efficientnetv2_s()
        elif opt.encoder == 'vssd':
            self.backbone_x = lib.Backbone_VMAMBA2()
            self.backbone_y = lib.Backbone_VMAMBA2()
        feature_channels = None,
        if opt.encoder == 'swin_tiny':
            feature_channels = [96, 192, 384, 768]
            embedding_dim = 192
        if opt.encoder == 'vssd':
            feature_channels = [96, 192, 384, 768]
            embedding_dim = 192
        if opt.encoder == 'mit_b2':
            feature_channels = [64, 128, 320, 512]
            embedding_dim = 128
        if opt.encoder == 'efficientnetV2':
            feature_channels = [48, 64, 160, 256]
            # feature_channels = [96, 192, 384, 768]
            embedding_dim = 64
        self.project1 = nn.Conv2d(96, 96, 1, 1)
        self.project2 = nn.Conv2d(192, 192, 1, 1)
        self.project3 = nn.Conv2d(384, 384, 1, 1)
        self.project4 = nn.Conv2d(768, 768, 1, 1)
        # self.mask2former = lib.MaskFormerModel()
        dims_decoder = feature_channels if feature_channels is not None else [64, 128, 256, 512]
        dims_decoder.reverse()
        self.mamba = lib.VMUNet(dims_decoder=dims_decoder)
        # self.vswin = lib.vswin_tiny()
        feature_channels.reverse()
        self.decode_head = lib.SegFormerHead(feature_channels, embedding_dim, self.bn, opt.seghead_dropout, 'bilinear', False)

        # self.fusion_module0 = lib.ContextSharingTransformer(feature_channels[0], feature_channels[0]*opt.ffn_dim_ratio, opt.fusion_module_dropout)
        # self.fusion_module1 = lib.ContextSharingTransformer(feature_channels[1], feature_channels[1]*opt.ffn_dim_ratio, opt.fusion_module_dropout)
        # self.fusion_module2 = lib.SemanticGatheringScatteringTransformer(feature_channels[2], feature_channels[2], opt.num_attn_heads, feature_channels[2]*opt.ffn_dim_ratio, opt.num_blocks, opt.fusion_module_dropout, opt.num_points, opt.threshold)
        # self.fusion_module3 = lib.SemanticGatheringScatteringTransformer(feature_channels[3], feature_channels[3], opt.num_attn_heads, feature_channels[3]*opt.ffn_dim_ratio, opt.num_blocks, opt.fusion_module_dropout, opt.num_points // 4, opt.threshold)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.init_backbone()

    def init_backbone(self):
        global saved_state_dict
        if self.opt.encoder == 'swin_tiny':
            saved_state_dict = torch.load('/home/qdk/code/MambaVOS-master/tools/pretrained_model/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        if self.opt.encoder == 'mit_b2':
            saved_state_dict = torch.load('/home/qdk/code/MambaVOS-master/tools/pretrained_model/mit_b2.pth')
        if self.opt.encoder == 'efficientnetV2':
            saved_state_dict = torch.load('/home/qdk/code/MambaVOS-master/tools/pretrained_model/pre_efficientnetv2-s.pth')
        if self.opt.encoder == 'vssd':
            saved_state_dict = torch.load('/home/qdk/code/MambaVOS-master/tools/pretrained_model/vssd_micro_best.pth')
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)
        if 'swin' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict['model'], strict=False)
            self.backbone_y.load_state_dict(saved_state_dict['model'], strict=False)
        elif 'mit' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)
        elif 'efficientnetV2' in self.opt.encoder:
            keys = list(saved_state_dict.keys())
            for key in keys:
                if 'head' in key:
                    del saved_state_dict[key]
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)
    def forward(self, x, y):
        # x_0, x_1, x_2, x_3 = self.backbone_x(x)
        # y_0, y_1, y_2, y_3 = self.backbone_y(y)
        output_x = self.backbone_x(x)
        output_y = self.backbone_y(y)
        fuse1 = self.project1(torch.concat([output_x[0], output_y[0]], 1))
        fuse2 = self.project2(torch.concat([output_x[1], output_y[1]], 1))
        fuse3 = self.project3(torch.concat([output_x[2], output_y[2]], 1))
        fuse4 = self.project4(torch.concat([output_x[3], output_y[3]], 1))
        # z_0 = self.fusion_module0(fuse1)
        # z_1 = self.fusion_module1(fuse2)
        # z_2 = self.fusion_module2(fuse3)
        # z_3 = self.fusion_module3(fuse4)
        # output_z1 = [z_0, z_1, z_2, z_3]
        # input_z = dict()
        # input_z['res1'] = fuse1
        # input_z['res2'] = fuse2
        # input_z['res3'] = fuse3
        # input_z['res4'] = fuse4
        #
        # output_z2 = self.mask2former(input_z)
        # z = self.decode_head(output_z2)
        z = [fuse1, fuse2, fuse3, fuse4]
        output_z = self.mamba(z)
        # output_z = self.vswin(z)
        output = self.decode_head(output_z)
        return output

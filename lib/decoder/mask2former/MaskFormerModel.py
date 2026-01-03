#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MaskFormerModel.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于DeformTransAtten的分割网络
'''

# here put the import lib
from torch import nn
from addict import Dict
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class MaskFormerHead(nn.Module):
    def __init__(self, conv_dim = 256):
        super().__init__()
        input_shape = get_input_shape()
        self.pixel_decoder = self.pixel_decoder_init(input_shape)
        self.predictor = self.predictor_init()

    def pixel_decoder_init(self, input_shape):
        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 0.0,
                                                 8,
                                                 1024,
                                                 6,
                                                 256,
                                                 256,
                                                 ["res2", "res3", "res4"],
                                                 4)
        return pixel_decoder

    def predictor_init(self):
        predictor = MultiScaleMaskedTransformerDecoder()
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
             features)
        # res2 = self.lateral_conv2(features['res2'])
        # res3 = self.lateral_conv3(features['res3'])
        # res4 = self.lateral_conv4(features['res4'])
        # res2 = self.output_conv2(res2)
        # res3 = self.output_conv3(res3)
        # res4 = self.output_conv4(res4)
        # multi_scale_features = [res4, res3, res2]
        # mask_features = self.lateral_conv1(features['res1'])
        # mask_features = self.output_conv1(mask_features)
        # predictions = {
        #     'pred_logits': multi_scale_features,
        #     'pred_masks': mask_features,
        # }
        predictions = self.predictor(multi_scale_features, mask_features, mask)

        return predictions

class MaskFormerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sem_seg_head = MaskFormerHead()
        self.conv1 = nn.Conv2d(in_channels=100, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=768, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=768, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        predictions = self.sem_seg_head(inputs)
        outputs = [self.conv1(predictions['pred_masks'])]
        outputs.append(self.conv2(predictions['pred_logits'][2]))
        outputs.append(self.conv3(predictions['pred_logits'][1]))
        outputs.append(self.conv4(predictions['pred_logits'][0]))
        return outputs


def get_input_shape():
    input_shape = dict()
    input_shape['res1'] = Dict({'channel': 96, 'stride': 4})
    input_shape['res2'] = Dict({'channel': 192, 'stride': 8})
    input_shape['res3'] = Dict({'channel': 384, 'stride': 16})
    input_shape['res4'] = Dict({'channel': 768, 'stride': 32})
    return input_shape

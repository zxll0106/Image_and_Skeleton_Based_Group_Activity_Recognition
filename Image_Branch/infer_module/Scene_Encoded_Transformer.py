import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils import calc_pairwise_distance_3d


class Scene_Encoded_Transformer_Per_Head(nn.Module):
    def __init__(self, num_features_context, NFB, K, N, layer_id, num_heads_per_layer, context_dropout_ratio = 0.1):
        super(Scene_Encoded_Transformer_Per_Head, self).__init__()
        self.num_features_context = num_features_context
        if layer_id == 1:
            self.downsample2 = nn.Conv2d(512, num_features_context, kernel_size = 1, stride=1)
            '''nn.init.kaiming_normal_(self.downsample1.weight)
            nn.init.kaiming_normal_(self.downsample2.weight)
            self.downsample = nn.Conv2d(D, num_features_context, kernel_size=1, stride=1)'''
            self.emb_roi = nn.Linear(NFB, num_features_context, bias=True)
        elif layer_id > 1:
            self.downsample = nn.Conv2d(512, num_features_context, kernel_size=1, stride=1)
            self.emb_roi = nn.Linear(num_features_context * num_heads_per_layer, num_features_context, bias=True)
            nn.init.kaiming_normal_(self.downsample.weight)
        self.N = N
        self.K = K
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(num_features_context)
        self.FFN = nn.Sequential(
            nn.Linear(num_features_context,num_features_context, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(num_features_context,num_features_context, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(num_features_context)
        self.att_map = None


    def forward(self, roi_feature, image_feature, layer_id = -1):
        """

        :param roi_feature:   # B*T*N, NFB
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        NFC = self.num_features_context
        BT, _,OH,OW = image_feature.shape
        K = self.K #roi_feature.shape[3]
        N = self.N #roi_feature.shape[0]//BT
        # assert N==12
        assert layer_id>=1
        if layer_id == 1:
            image_feature = self.downsample2(image_feature)
            emb_roi_feature = self.emb_roi(roi_feature) # B*T*N, D
        elif layer_id > 1:
            emb_roi_feature = self.emb_roi(roi_feature)
            image_feature = self.downsample(image_feature)
        emb_roi_feature = emb_roi_feature.reshape(BT, N, 1, 1, NFC) # B*T, N, 1, 1, D
        image_feature = image_feature.reshape(BT, 1, NFC, OH, OW) # B*T, 1, D, OH, OW
        image_feature = image_feature.transpose(2,3) # B*T, 1, OH, D, OW

        a = torch.matmul(emb_roi_feature, image_feature) # B*T, N, OH, 1, OW
        a = a.reshape(BT, N, -1) # B*T, N, OH*OW
        A = F.softmax(a, dim=2)  # B*T, N, OH*OW
        self.att_map = A
        image_feature = image_feature.transpose(3,4).reshape(BT, OH*OW, NFC)

        context_encoding_roi = self.dropout(torch.matmul(A, image_feature).reshape(BT*N, NFC))
        emb_roi_feature = emb_roi_feature.reshape(BT*N, NFC)
        context_encoding_roi = self.layernorm1(context_encoding_roi + emb_roi_feature)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi)
        return context_encoding_roi


class Scene_Encoded_Transformer(nn.Module):
    def __init__(self, num_heads_per_layer, num_layers, num_features_context, NFB, K, N, context_dropout_ratio=0.1):
        super(Scene_Encoded_Transformer, self).__init__()
        self.CET = nn.ModuleList()
        for i in range(num_layers):
            for j in range(num_heads_per_layer):
                self.CET.append(Scene_Encoded_Transformer_Per_Head(num_features_context, NFB, K, N, i+1, num_heads_per_layer, context_dropout_ratio))
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.vis_att_map = torch.empty((0, 12, 43 * 78), dtype = torch.float32)

    def forward(self, roi_feature, image_feature):
        """
        :param roi_feature:   # B*T*N, NFB,
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        for i in range(self.num_layers):
            MHL_context_encoding_roi= []
            for j in range(self.num_heads_per_layer):
                MHL_context_encoding_roi.append(self.CET[i*self.num_heads_per_layer + j](roi_feature, image_feature,i+1))
            roi_feature = torch.cat(MHL_context_encoding_roi, dim=1)

        return roi_feature




if __name__=='__main__':
    '''test SpatialMessagePassing
    s = SpatialMessagePassing(4, 4)
    t = torch.rand(1,4,4)
    mask = torch.ones((1,4,4))
    print(s(t, mask))
    print(t)'''

    '''test Pose2d_Encoder
    cfg = Config('volleyball')
    p2d = Pose2d_Encoder(cfg)'''

    '''test Context Encoding Transformer
    cet = ContextEncodingTransformer(num_features_context=128,D=256, K=5, N=12, layer_id=1,
                                     num_heads_per_layer=1, context_dropout_ratio = 0.1)
    roi_feature = torch.rand(36,256,5,5)
    image_feature = torch.rand(3, 256, 45, 80)
    context_encoding_roi = cet(roi_feature, image_feature, 1)
    print(context_encoding_roi.shape)'''


    '''test multi-layer multi-head context encoding transformer'''
    mlhcet = MultiHeadLayerContextEncoding(3, 1, num_features_context=128,  D=256, K=5, N=12, context_dropout_ratio=0.1)
    roi_feature = torch.rand(36, 256, 5, 5)
    image_feature = torch.rand(3, 256, 45, 80)
    context_encoding_roi = mlhcet(roi_feature, image_feature)
    print(context_encoding_roi.shape)

    '''test temporal message passing
    tmp =  multiheadTemporalMessage(128, 128, 3)
    t1 = torch.rand(6,12,128)
    mask = generate_temporal_mask(2, 12, 3)
    print(mask.shape)
    output = tmp(t1, mask, shortcut_connection=True)
    print(output)
    print(output.shape)'''

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from utils import calc_pairwise_distance_3d
import math

class SpatialTransformer(nn.Module):
    def __init__(self, context_dim, context_dropout_ratio = 0.1):
        super(SpatialTransformer, self).__init__()
        self.context_dim = context_dim

        self.Wq = nn.Linear(context_dim,context_dim)
        self.Wk = nn.Linear(context_dim, context_dim)
        self.Wv = nn.Linear(context_dim, context_dim)
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(context_dim)
        self.FFN = nn.Sequential(
            nn.Linear(context_dim, context_dim, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(context_dim, context_dim, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(context_dim)
        self.att_map = None


    def forward(self,boxes_features,mask=None):
        """

        :param boxes_features:   # # B,T,N, NFC
        :return:
        """
        NFC = self.context_dim
        B,T,N,_ = boxes_features.shape

        q = self.Wq(boxes_features) # B,T,N,context_dim
        k = self.Wk(boxes_features) # B,T,N,context_dim
        v=self.Wv(boxes_features) # B,T,N,context_dim
        q = q.reshape(B*T, N, NFC)
        k = k.reshape(B*T, N, NFC)
        v = v.reshape(B*T, N, NFC)

        a = torch.matmul(q, k.transpose(1,2)) # B*T, N, N
        a=a/math.sqrt(NFC)

        if mask is not None:
            a=a.masked_fill_(mask,-1e9)

        A = F.softmax(a, dim=2)

        context_encoding_roi = self.dropout(torch.matmul(A, v)) #B*T,N,NFC
        context_encoding_roi = self.layernorm1(context_encoding_roi + q)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi) #B*T,N,NFC
        context_encoding_roi=context_encoding_roi.reshape(B,T,N,NFC)
        return context_encoding_roi

class TemporalTransformer(nn.Module):
    def __init__(self, context_dim, context_dropout_ratio = 0.1):
        super(TemporalTransformer, self).__init__()
        self.context_dim = context_dim

        self.Wq = nn.Linear(context_dim,context_dim)
        self.Wk = nn.Linear(context_dim, context_dim)
        self.Wv = nn.Linear(context_dim, context_dim)
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(context_dim)
        self.FFN = nn.Sequential(
            nn.Linear(context_dim, context_dim, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(context_dim, context_dim, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(context_dim)
        self.att_map = None


    def forward(self,boxes_features):
        """

        :param boxes_features:   # B,T,N, NFC
        :return:
        """
        NFC = self.context_dim
        B,T,N,_ = boxes_features.shape

        boxes_features=boxes_features.transpose(1,2)
        q = self.Wq(boxes_features) # B,N,T,context_dim
        k = self.Wk(boxes_features) # B,N,T,context_dim
        v=self.Wv(boxes_features) # B,N,T,context_dim
        q = q.reshape(B*N, T, NFC)
        k = k.reshape(B*N, T, NFC)
        v = v.reshape(B*N, T, NFC)

        a = torch.matmul(q, k.transpose(1,2)) # B*N,T,T
        a=a/math.sqrt(NFC)
        A = F.softmax(a, dim=2)

        context_encoding_roi = self.dropout(torch.matmul(A, v)) #B*N,T,NFC
        context_encoding_roi = self.layernorm1(context_encoding_roi + q)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi) #B*N,T,NFC
        context_encoding_roi=context_encoding_roi.reshape(B,N,T,NFC)
        context_encoding_roi=context_encoding_roi.transpose(1,2)
        return context_encoding_roi

class Spatial_Temporal_Transformer(nn.Module):
    def __init__(self, context_dim, context_dropout_ratio = 0.1):
        super(Spatial_Temporal_Transformer, self).__init__()
        self.context_dim = context_dim

        self.Wq = nn.Linear(context_dim, context_dim)
        self.Wk = nn.Linear(context_dim, context_dim)
        self.Wv = nn.Linear(context_dim, context_dim)
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(context_dim)
        self.FFN = nn.Sequential(
            nn.Linear(context_dim, context_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(context_dim, context_dim, bias=True)
        )
        self.layernorm2 = nn.LayerNorm(context_dim)
        self.att_map = None

    def forward(self,spatial_features,temporal_features):
        """

                :param spatial_features: B,T,N, NFC temporal_features: B,T,N, NFC
                :return:
                """
        NFC = self.context_dim
        B, T, N, _ = spatial_features.shape

        spatial_features=spatial_features.reshape(B,T*N,NFC)
        temporal_features=temporal_features.reshape(B,T*N,NFC)

        q = spatial_features  # B,T*N,context_dim
        k = temporal_features  # B,T*N,context_dim
        v = temporal_features  # B,T*N,context_dim

        a = torch.matmul(q, k.transpose(1, 2))  # B,T*N,T*N
        a = a / math.sqrt(NFC)
        A = F.softmax(a, dim=2)

        context_encoding_roi = self.dropout(torch.matmul(A, v))  # B,T*N,NFC
        context_encoding_roi = self.layernorm1(context_encoding_roi + q)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi)  # B,T*N,NFC
        context_encoding_roi = context_encoding_roi.reshape(B, T, N, NFC)
        return context_encoding_roi


if __name__ == '__main__':
    boxes_feature=torch.randn((2,10,12,1536))
    spatial_transformer=TemporalTransformer(1536)
    contetx_encoding_roi=spatial_transformer(boxes_feature)
    print(boxes_feature.shape)
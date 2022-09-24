import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from st_gcn import Model
from collections import OrderedDict

class Pose_Encoder_collective(nn.Module):
    def __init__(self):
        super(Pose_Encoder_collective, self).__init__()

        # self.st_gcn=Model(in_channels=3,num_class=400,graph_args={'layout':'openpose','strategy':'spatial'},edge_importance_weighting=True)
        self.st_gcn = Model(in_channels=3, num_class=400, graph_args={'layout': 'openpose', 'strategy': 'spatial'},
                            edge_importance_weighting=True)
        self.fc_actions = nn.Conv2d(256, 5, kernel_size=1)
        self.fc_activities=nn.Conv2d(256, 4, kernel_size=1)

    def loadmodel(self,weights_path):
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v] for k, v in weights.items()])
        self.st_gcn.load_state_dict(weights)
        print('Load model states from: ',weights_path)
        # for name, parameters in self.st_gcn.named_parameters():
        #     print(name, ':', parameters)
        # for name, parameters in self.fc_actions.named_parameters():
        #     print(name, ':', parameters)
        # for name, parameters in self.fc_activities.named_parameters():
        #     print(name, ':', parameters)

    def forward(self,keypoints,bboxes_num_in):
        keypoints=keypoints.permute(0,4,1,3,2) #B,3,T,18,N

        B=keypoints.shape[0]
        T=keypoints.shape[2]


        bboxes_num_in = bboxes_num_in.reshape(B, T)

        action_scores=[]
        activity_scores=[]

        keypoints_N=[]
        for b in range(B):
            N=bboxes_num_in[b][0]

            individual_features=[]

            keypoints_b = keypoints[b, :, :, :, :N] #3,T,18,N
            keypoints_b=keypoints_b.unsqueeze(0) #1,3,T,18,N

            keypoints_N.append(keypoints_b)

        keypoints_N=torch.cat(keypoints_N,dim=-1) #1,3,T,18,ALL_N

        features_all=self.st_gcn(keypoints_N) #ALL_N*1,256,1,1

        individual_scores=self.fc_actions(features_all).reshape(-1,5) #ALL_N,5
        action_scores = individual_scores  # ALL_N,5

        for b in range(B):
            N=bboxes_num_in[b][0]
            if b==0:
                beg=0
            else:
                beg=bboxes_num_in[b-1][0]
            group_features, _ = torch.max(features_all[beg:beg+N,:,:,:], dim=0) #256,1,1
            group_features = group_features.unsqueeze(0)  # 1,256,1,1

            # group_features_path='group_features_9176.pt'
            # if os.path.exists(group_features_path):
            #     original=torch.load(group_features_path)
            #     original.append(group_features.reshape(1,-1))
            #     torch.save(original,group_features_path)
            # else:
            #     torch.save([group_features.reshape(1,-1)], group_features_path)

            group_scores = self.fc_activities(group_features).reshape(-1, 4)  # 1,4
            activity_scores.append(group_scores)

        activity_scores=torch.cat(activity_scores,dim=0) #B,4

        return action_scores,activity_scores


class Pose_Encoder_volleyball(nn.Module):
    def __init__(self):
        super(Pose_Encoder_volleyball, self).__init__()

        self.st_gcn=Model(in_channels=3,num_class=400,graph_args={'layout':'openpose','strategy':'spatial'},edge_importance_weighting=True)
        # self.transformer=SpatialTransformer(context_dim=256,context_dropout_ratio=0.3)
        self.fc_actions = nn.Conv2d(256, 9, kernel_size=1)
        self.fc_activities=nn.Conv2d(256, 8, kernel_size=1)

    def loadmodel(self,weights_path):
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v] for k, v in weights.items()])
        self.st_gcn.load_state_dict(weights)
        print('Load model states from: ',weights_path)
        # for name, parameters in self.st_gcn.named_parameters():
        #     print(name, ':', parameters)
        # for name, parameters in self.fc_actions.named_parameters():
        #     print(name, ':', parameters)
        # for name, parameters in self.fc_activities.named_parameters():
        #     print(name, ':', parameters)

    def forward(self,keypoints):
        keypoints = keypoints.permute(0, 4, 1, 3, 2)  # B,3,T,18,N

        B = keypoints.shape[0]
        T = keypoints.shape[2]

        individual_features = self.st_gcn(keypoints)  # B*N,256,1,1

        action_scores = self.fc_actions(individual_features).reshape(B * 12, -1)  # B*N,9,1,1

        group_features, _ = torch.max(individual_features.reshape(B, 12, 256, 1, 1), dim=1)  # B,256,1,1
        activity_scores = self.fc_activities(group_features).reshape(B, -1)  # B,8,1,1

        return {'actions': action_scores, 'activities': activity_scores}
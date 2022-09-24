from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from infer_module.Scene_Encoded_Transformer import Scene_Encoded_Transformer
from infer_module.positional_encoding import Context_PositionEmbeddingSine, Embfeature_PositionEmbedding, TemporalPositionalEncoding
from infer_module.Spatial_Temporal_Transformer import SpatialTransformer, TemporalTransformer, Spatial_Temporal_Transformer
import collections


class Model_collective(nn.Module):
    def __init__(self, cfg):
        super(Model_collective, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_features_context = 128

        T, N = cfg.num_frames, cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        else:
            assert False
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        #self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        self.Scene_Encoded_Transformer = \
            Scene_Encoded_Transformer(
                num_heads_context, 1,
                num_features_context, NFB, K, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16,512//2)
        self.embfeature_positionembedding=Embfeature_PositionEmbedding(cfg=self.cfg)
        context_dim = in_dim + num_heads_context * num_features_context

        self.temporal_pe=TemporalPositionalEncoding(d_model=context_dim,dropout=self.cfg.train_dropout_prob)
        self.spatial_encoder1 = SpatialTransformer(context_dim=context_dim,
                                                   context_dropout_ratio=self.cfg.train_dropout_prob)
        self.temporal_encoder1 = TemporalTransformer(context_dim=context_dim,
                                                     context_dropout_ratio=self.cfg.train_dropout_prob)

        self.fusion_layer = nn.Linear(2 * context_dim, context_dim)


        self.dpi_nl = nn.LayerNorm([context_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(context_dim, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    #         nn.init.zeros_(self.fc_gcn_3.weight)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4
        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, -1)  # B*T,MAX_N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B, T,MAX_N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all)

        if self.cfg.lite_dim:
            boxes_features_all = boxes_features_all.permute(0, 3, 1, 2)
            boxes_features_all = self.point_conv(boxes_features_all)
            boxes_features_all = boxes_features_all.permute(0, 2, 3, 1)
            boxes_features_all = self.point_ln(boxes_features_all)
            boxes_features_all = F.relu(boxes_features_all, inplace = True)
        else:
            None

        context = outputs[-1]
        context=context.reshape(B,T,-1,OH,OW)
        boxes_in_flat=boxes_in_flat.reshape(B,T,MAX_N,4)

        # boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, NFB)
        # boxes_in = boxes_in.reshape(B, T, MAX_N, 4)

        #actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :N, :].reshape(1, T, N, -1)  # T,N,NFB
            boxes_in_flat_b=boxes_in_flat[b,:,:N,:].reshape(T*N,-1)
            boxes_features=boxes_features.reshape(T*N,-1)
            boxes_features=self.embfeature_positionembedding(boxes_features,boxes_in_flat_b)
            context_b=context[b,:,:,:,:] #T,D,OH,OW
            context_b=self.context_positionembedding1(context_b)
            context_states=self.Scene_Encoded_Transformer(boxes_features,context_b,N)
            boxes_features = torch.cat((boxes_features, context_states), dim=1)
            boxes_features=boxes_features.reshape(T,N,-1)

            spatial_boxes_features=boxes_features
            spatial_boxes_features=self.spatial_encoder1(spatial_boxes_features.unsqueeze(0)).squeeze(0)
            temporal_boxes_features=boxes_features.transpose(0,1)
            temporal_boxes_features=self.temporal_pe(temporal_boxes_features).reshape(N,T,-1)
            temporal_boxes_features=temporal_boxes_features.transpose(0,1)
            temporal_boxes_features=self.temporal_encoder1(temporal_boxes_features.unsqueeze(0)).squeeze(0)

            ST_boxes_features=torch.cat((spatial_boxes_features,temporal_boxes_features),dim=2)
            ST_boxes_features=self.fusion_layer(ST_boxes_features)

            if self.cfg.backbone == 'res18':
                graph_boxes_features = ST_boxes_features.reshape( T, N, -1)
                graph_boxes_features = self.dpi_nl(graph_boxes_features)
                graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
                boxes_features = boxes_features.reshape(T, N, -1)
                boxes_states = graph_boxes_features + boxes_features
                boxes_states = self.dropout_global(boxes_states)
            elif self.cfg.backbone == 'vgg16' or self.cfg.backbone == 'inv3':
                graph_boxes_features = ST_boxes_features.reshape(T, N, -1)
                boxes_features = boxes_features.reshape(T, N, -1)
                boxes_states = graph_boxes_features + boxes_features
                boxes_states = self.dpi_nl(boxes_states)
                boxes_states = F.relu(boxes_states, inplace=True)
                boxes_states = self.dropout_global(boxes_states)


            NFS = NFG
            # boxes_states = boxes_states.view(T, N, -1)

            # Predict actions
            # actn_score = self.fc_actions(boxes_states)  # T,N, actn_num
            # actn_score = torch.mean(actn_score, dim=0).reshape(N, -1)  # N, actn_num
            # actions_scores.append(actn_score)
            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim = 1)  # T, NFS
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

        # actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return activities_scores # activities_scores # actions_scores,


class Model_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(Model_volleyball, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_features_context = 128

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        # self.gcn_list = torch.nn.ModuleList([ GCN_Module(self.cfg)  for i in range(self.cfg.gcn_layers) ])
        if self.cfg.lite_dim:
            in_dim = self.cfg.lite_dim
            print_log(cfg.log_path, 'Activate lite model inference.')
        else:
            in_dim = NFB
            print_log(cfg.log_path, 'Deactivate lite model inference.')

        # TCE Module Loading
        self.Scene_Encoded_Transformer = \
            Scene_Encoded_Transformer(
                num_heads_context, 1,
                num_features_context, NFB, K, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16, 512 / 2)
        self.embfeature_positionembedding = Embfeature_PositionEmbedding(cfg=cfg, num_pos_feats=NFB // 2)




        # DIN
        context_dim = in_dim + num_heads_context * num_features_context

        self.temporal_pe1=TemporalPositionalEncoding(d_model=context_dim,dropout=self.cfg.train_dropout_prob)
        self.temporal_pe2 = TemporalPositionalEncoding(d_model=context_dim, dropout=self.cfg.train_dropout_prob)
        self.spatial_encoder1 = SpatialTransformer(context_dim=context_dim,context_dropout_ratio=self.cfg.train_dropout_prob)
        # self.spatial_encoder2=SpatialTransformer(context_dim=context_dim,context_dropout_ratio=self.cfg.train_dropout_prob)
        self.temporal_encoder1=TemporalTransformer(context_dim=context_dim,context_dropout_ratio=self.cfg.train_dropout_prob)
        # self.temporal_encoder2=TemporalTransformer(context_dim=context_dim,context_dropout_ratio=self.cfg.train_dropout_prob)

        self.fusion_layer1=nn.Linear(2*context_dim,context_dim)
        self.fusion_layer2=nn.Linear(2*context_dim,context_dim)
        self.dropout1=nn.Dropout(p=self.cfg.train_dropout_prob)
        self.dropout2=nn.Dropout(p=self.cfg.train_dropout_prob)

        self.dpi_nl = nn.LayerNorm([T, N, context_dim])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # Lite Dynamic inference
        if self.cfg.lite_dim:
            self.point_conv = nn.Conv2d(NFB, in_dim, kernel_size=1, stride=1)
            self.point_ln = nn.LayerNorm([T, N, in_dim])
            self.fc_activities = nn.Linear(in_dim, self.cfg.num_activities)
        else:
            self.fc_activities = nn.Linear(context_dim, self.cfg.num_activities)
            self.fc_actions=nn.Linear(context_dim,self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        pos_threshold = self.cfg.pos_threshold

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        if self.cfg.lite_dim:
            boxes_features = boxes_features.permute(0, 3, 1, 2)
            boxes_features = self.point_conv(boxes_features)
            boxes_features = boxes_features.permute(0, 2, 3, 1)
            boxes_features = self.point_ln(boxes_features)
            boxes_features = F.relu(boxes_features, inplace=True)
        else:
            None

        # Context Positional Encoding
        context = outputs[-1]
        context = self.context_positionembedding1(context)
        # context=context.reshape(B*T,-1,OH,OW)

        # boxes_features=self.embfeature_positionembedding(boxes_features.reshape(B*T*N,-1),boxes_in_flat)

        # Embedded Feature Context Encoding
        context_states = self.Scene_Encoded_Transformer(boxes_features, context)
        context_states = context_states.reshape(B, T, N, -1)
        boxes_features=boxes_features.reshape(B,T,N,-1)
        boxes_features = torch.cat((boxes_features, context_states), dim=3)

        spatial_boxes_features=boxes_features
        spatial_boxes_features = self.spatial_encoder1(spatial_boxes_features)

        temporal_boxes_features = boxes_features.transpose(1, 2)
        temporal_boxes_features = temporal_boxes_features.reshape(B * N, T, -1)
        temporal_boxes_features = self.temporal_pe1(temporal_boxes_features).reshape(B, N, T, -1)
        temporal_boxes_features = temporal_boxes_features.transpose(1, 2)
        temporal_boxes_features = self.temporal_encoder1(temporal_boxes_features)  # B,T,N,NFC

        ST_boxes_features = torch.cat((spatial_boxes_features, temporal_boxes_features), dim=3)
        ST_boxes_features = self.fusion_layer1(ST_boxes_features)  # B,T,N,-1


        if self.cfg.backbone == 'res18':
            graph_boxes_features = ST_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.dpi_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
        elif self.cfg.backbone == 'vgg16':
            graph_boxes_features = ST_boxes_features.reshape(B, T, N, -1)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dpi_nl(boxes_states)
            boxes_states = F.relu(boxes_states, inplace=True)
            boxes_states = self.dropout_global(boxes_states)

        # Predict actions
        boxes_states_flat=boxes_states.reshape(-1,boxes_states.shape[-1])  #B*T*N, NFS
        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num


        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        actions_scores = actions_scores.reshape(B,T,N,-1)
        actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        return {'activities': activities_scores}  # actions_scores, activities_scores
import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('collective')
cfg.inference_module_name = 'collective'

cfg.device_list="0"
cfg.training_stage=2
cfg.use_gpu = True
cfg.use_multi_gpu = False
cfg.train_backbone = True
cfg.load_backbone_stage2 = True

# VGG16
cfg.backbone = 'vgg16'
cfg.image_size = 480, 720
cfg.out_size = 15, 22
cfg.emb_features = 512
cfg.stage1_model_path = ''
cfg.load_backbone_stage2=False

cfg.num_boxes = 13
cfg.num_actions = 5
cfg.num_activities = 4
cfg.num_frames = 10
cfg.num_graph = 4
cfg.tau_sqrt=True
cfg.batch_size = 2
cfg.test_batch_size = 2
cfg.test_interval_epoch = 1
cfg.train_learning_rate = 5e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-4
cfg.lr_plan = {}
cfg.max_epoch = 30

cfg.train_dropout_prob = 0.3

cfg.exp_note='collective'
train_net(cfg)
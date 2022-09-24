import sys
sys.path.append(".")
from train_net import *
from config_arg import Config

cfg=Config('collective')

cfg.backbone = 'vgg16'
cfg.image_size = 480, 720
cfg.out_size = 15, 22
cfg.emb_features = 512
cfg.stage1_model_path = ''

cfg.device_list="0"
cfg.training_stage=2
cfg.st_gcn_model_path='st_gcn.kinetics.pt'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.test_before_train=False

cfg.num_boxes=13
cfg.num_actions=5
cfg.num_activities=4
cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=2
cfg.test_batch_size=2
cfg.test_interval_epoch = 1
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=50

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

# Dynamic Inference setup
cfg.group = 1
cfg.stride = 1
cfg.ST_kernel_size = (3, 3)
cfg.dynamic_sampling = True
cfg.sampling_ratio = [1]  # [1,2,4]
cfg.lite_dim = None # 128
cfg.scale_factor = True
cfg.beta_factor = False
cfg.hierarchical_inference = False
cfg.parallel_inference = False
cfg.num_DIM = 1
cfg.train_dropout_prob = 0.3

cfg.use_multi_gpu=False

cfg.exp_note='Collective_stage2'
train_net(cfg)
import sys
sys.path.append(".")
from train_net import *
from config_arg import Config

cfg=Config('volleyball')

cfg.device_list="0"
cfg.training_stage=2
cfg.stage1_model_path=''  #PATH OF THE BASE MODEL
cfg.st_gcn_model_path='st_gcn.kinetics.pt'
cfg.train_backbone=False
cfg.test_before_train=False

cfg.batch_size=2 #32
cfg.test_batch_size=2
cfg.num_actions=9
cfg.num_activities=8
cfg.num_frames=10
cfg.train_learning_rate=2e-4
cfg.max_epoch=50
cfg.actions_weights=[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]

cfg.exp_note='Volleyball_stage2'
train_net(cfg)
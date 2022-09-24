# Image Branch of Spatial Temporal Network for Image and Skeleton Based Group Activity Recognition

This repo contains code of our paper "Spatial Temporal Network for Image and Skeleton Based Group Activity Recognition." 

_Xiaolin Zhai, Zhengxi Hu, Dingye Yang, Lei Zhou and JingTai Liu_
        


## Dependencies

- Software Environment: Linux 
- Hardware Environment: NVIDIA RTX 3090
- Python `3.8`
- PyTorch `1.11.0`, Torchvision `0.12.0`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)



## Prepare Datasets

1. Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
2. Unzip the dataset file into `data/volleyball` or `data/collective`.


## Train
1. **Train the Base Model**: Fine-tune the base model for the dataset. 
```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
  ```

2. **Train the Image Branch**:
```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage2.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage2.py
  ```
 
# Skeleton Branch of Spatial Temporal Network for Image and Skeleton Based Group Activity Recognition

This repo contains code of our paper "Spatial Temporal Network for Image and Skeleton Based Group Activity Recognition." 

_Xiaolin Zhai, Zhengxi Hu, Dingye Yang, Lei Zhou and JingTai Liu_
        


## Dependencies

- Software Environment: Linux 
- Hardware Environment: NVIDIA RTX 3090
- Python `3.8`
- PyTorch `1.11.0`, Torchvision `0.12.0`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)



## Extended Skeleton Data

Unzip `skeleton_data_volleyball.zip` in `data/volleyball` and `skeleton_data_collective.zip` in `data/collective`.


## Train
 **Train the Skeleton Branch**: 
```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective.py
  ```


 
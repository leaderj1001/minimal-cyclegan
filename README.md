# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks using Pytorch
## Reference
 - [Paper Link](https://arxiv.org/abs/1703.10593)
 - Author: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
 - Organization: Berkeley AI Research (BAIR) Lab, UC Berkeley
 - 

## Prepare data
```
dataset
 ├── monet2photo
 │   ├── trainA
 │   ├── trainB
 |   ├── testA
 │   ├── testB
```

## Usage
  - train
  ```
  python main.py
  ```
  
  - test
  ```
  python main.py --training True --evaluation True
  ```
  
## Results

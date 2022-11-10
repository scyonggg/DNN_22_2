# Quick Start

0. Requirements
  ```
  timm
  wandb
  pytorch (버전 상관 X, 아마도)
  ```
1. `conf/settings.py`에서 DATA_PATH를 train dataset 위치로 수정
2. wandb를 사용할 경우

    a. `script.sh`에 --wandb를 추가
  
    b. 터미널에 wandb login 입력 후, wandb 홈페이지에서 API Keys를 복-붙
  
3. script에 옵션들 확인 후 `bash script.sh`
4. wandb에 잘 올라오는지 확인

## To DO
1. 도커 업데이트 or 콘다 업데이트
2. requirement 업로드

$ CUDA_VISIBLE_DEVICES=0 ~~~~
gpu 선택해서돌리기

---

# Bag of Tricks for Image Classification with Convolutional Neural Networks 


This repo was inspired by Paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

I would test popular training tricks as many as I can for improving image classification accuarcy, feel
free to leave a comment about the tricks you want me to test(please write the referenced paper along with
the tricks)

## hardware
Using 4 Tesla P40 to run the experiments

## dataset

I will use [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset instead of ImageNet,
just for simplicity, this is a fine-grained image classification dataset, which contains 200 birds categlories, 
5K+ training images, and 5K+ test images.The state of the art acc on vgg16 is around 73%(please correct me if 
I was wrong).You could easily change it to the ones you like: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/), [Stanford Cars](http://vision.stanford.edu/aditya86/ImageNetDogs/).
Or even ImageNet.

## network

Use a VGG16 network to test my tricks, also for simplicity reasons, since VGG16 is easy to implement. I'm considering
switch to AlexNet, to see how powerful these tricks are.

## tricks

tricks I've tested, some of them were from the Paper [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) :

|trick|referenced paper|
|:---:|:---:|
|xavier init|[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)|
|warmup training|[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677v2)|
|no bias decay|[Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205vx)|
|label smoothing|[Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567v3))|
|random erasing|[Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2)|
|cutout|[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2)|
|linear scaling learning rate|[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677v2)|
|cosine learning rate decay|[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)|

**and more to come......**


## result

baseline(training from sctrach, no ImageNet pretrain weights are used): 

vgg16 64.60% on [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset, lr=0.01, batchsize=64

effects of stacking tricks 

|trick|acc|
|:---:|:---:|
|baseline|64.60%|
|+xavier init and warmup training|66.07%|
|+no bias decay|70.14%|
|+label smoothing|71.20%|
|+random erasing|does not work, drops about 4 points|
|+linear scaling learning rate(batchsize 256, lr 0.04)|71.21%|
|+cutout|does not work, drops about 1 point|
|+cosine learning rate decay|does not work, drops about 1 point|

# Differential Privacy
Brock University, 2023
Sager Kudrick's Differential Privacy project for Professor Renata Queiroz Dividino

# Introduction

This project contains four main files:
1. imdb_train.py - responsible for training an imdb sentiment prediction model
2. mnist_train.py - responsible for training a handwriting prediction model (for numbers 0-9)
3. imdb_evaluate.py - responsible for evaluating reviews stores in imdb_samples/reviews.json
4. imnist_evaluate.py - responsible for evaluating handwriting in real time, allowing the user to draw and instantly evaluate an image

# Installation
Using Opacus with CUDA requires specific PyTorch and CUDA installations. If you don't have a compatible GPU, you can skip these steps and continue on from "Training Models", and use the CPU command. 

# Installing PyTorch

1. Choose your PyTorch Build, OS, Package, Language, and CUDA 11.7 from https://pytorch.org/get-started/locally/
2. Paste the built command to install the appropriate versions. An example installation for Stable (2.0.0), Windows, Pip, Python, CUDA 11.7 looks like this: pip3 "install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"

# Installing CUDA (Not required, but recommended) 

Because we require CUDA 11.7 or CUDA 11.8, we need to use previous versions. CUDA 11.7 is available from:
https://developer.nvidia.com/cuda-11-7-0-download-archive

# Training models

Both the IMDb & MNIST trainers are capaable of training with and without differential privacy. Having CUDA installed is recommended for quicker train times, but not required. See below for instructions on training models.

# Training commands

Training without CUDA:
```
python imdb_train.py --device cpu
```

Training with CUDA - this is default, but to explicitly run with cuda: 
```
python imdb_train.py --device cuda:0
```

# MNIST prediction with & without differential privacy
<p align="center">
  <img width="500" height="500" src="https://github.com/SagerKudrick/ml-dp/blob/main/Pictures/mnist_predictions.gif">
</p>

# Setting it up

 # IMDb predictions with & without differential privacy
<p align="center">
  
  <img width="319" height="553" src="https://github.com/SagerKudrick/ml-dp/blob/main/Pictures/imdb_prediction_results.PNG">
</p>


# <p align="center">Protecting Privacy in Machine Learning: Training Models with Differential Privacy & Opacus
Sager Kudrick</p>  
5919170  
Professor Renata Queiroz Dividino  
Brock University, 2023  

# Table Of Contents
1. Introduction
2. Pre-Bundled Models
3. Installation
   1. Installing PyTorch
   2. CUDA compatible GPUs
   3. Installing CUDA (Not required, but recommended)
4. Training models
   1. Training commands
   2. Training with & without differential privacy
5. MNIST prediction with & without differential privacy
6. IMDb predictions with & without differential privacy
7. Citations

# Introduction

This project allows you to train two different models with & without differential privacy: MNIST, and IMDb. It then allows you to evaluate custom reviews, or custom handwritten images with the trained models. Pre-trained models are also provided.

This project contains four main files:
1. imdb_train.py - responsible for training an imdb sentiment prediction model
2. mnist_train.py - responsible for training a handwriting prediction model (for numbers 0-9)
3. imdb_evaluate.py - responsible for evaluating reviews stores in imdb_samples/reviews.json
4. imnist_evaluate.py - responsible for evaluating handwriting in real time, allowing the user to draw and instantly evaluate an image

# Pre-Bundled Models
Rather than training your own models, you can instead use pre-trained models packaged with this project. They are listed as such:

1. models/mnist_with_dp.pt - an MNIST model trained with differential privacy.
2. models/mnist_no_dp.pt - an MNIST model trained WITHOUT differential privacy.
3. models/imdb_with_dp.pt - an IMDb sentiment prediction model trained with differential privacy.
4. models/mnist_with_dp.pt - an IMDb sentiment prediction model trained WITHOUT differential privacy.

# Installation
Using Opacus with CUDA requires specific PyTorch and CUDA installations. If you don't have a compatible GPU, you can skip these steps and continue on from "Training Models", and use the CPU command. 

# Installing PyTorch

1. Choose your PyTorch Build, OS, Package, Language, and CUDA 11.7 from https://pytorch.org/get-started/locally/
2. Paste the built command to install the appropriate versions. An example installation for Stable (2.0.0), Windows, Pip, Python, CUDA 11.7 looks like this: pip3 "install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"

# CUDA compatible GPUs 
A list of compatible GPUs are listed on https://developer.nvidia.com/cuda-gpus - please ensure you have a compatible GPU before trying to install or train with CUDA. 

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

# Training with & without differential privacy

By default the trainers will train with differential privacy. 

To train without differential privacy:
```
python imdb_train.py --disable-dp
```

# MNIST prediction with & without differential privacy
<p align="center">
  <img width="500" height="500" src="https://github.com/SagerKudrick/ml-dp/blob/main/Pictures/mnist_predictions.gif">
</p>


 # IMDb predictions with & without differential privacy
<p align="center">
  
  <img width="319" height="553" src="https://github.com/SagerKudrick/ml-dp/blob/main/Pictures/imdb_prediction_results.PNG">
</p>

# Citations
1. Near, J. P., & Abuah, C. (2021). Programming Differential Privacy (Vol. 1). Retrieved from https://uvm-plaid.github.io/programming-dp/

2. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407. doi: 10.1561/0400000042.

3. Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., & Mironov, I. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv preprint arXiv:2109.12298.

4. Fridman, L., & Trask, A. (2018, October 31). Introduction to Deep Learning [Video]. YouTube. https://www.youtube.com/watch?v=4zrU54VIK6k&ab_channel=LexFridman

5. Opacus. (n.d.). Tutorials [Webpage]. Retrieved March 6th, 2023, from https://opacus.ai/tutorials/

6. Mironov, I., Talwar, K., & Zhang, L. (2019). R'enyi Differential Privacy of the Sampled Gaussian Mechanism. arXiv preprint arXiv:1908.10530. https://arxiv.org/abs/1908.10530

7. Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., & Mironov, I. (2022). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv preprint arXiv:2109.12298. https://arxiv.org/abs/2109.12298


Citation if you use Opacus in papers
```
@article{opacus,
  title={Opacus: {U}ser-Friendly Differential Privacy Library in {PyTorch}},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```


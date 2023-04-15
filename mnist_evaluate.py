#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
import torch.nn.functional as F
from enum import Enum
from ml_args import get_args 
import pygame as pyg
import io
import base64
import draw_image as draw_img
import win32api

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# MNIST ml model 
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "MNIST model"

transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

def transform_image(image_url):

    # open the image
    #image = Image.open(image_url).convert('L')
    image_url = image_url.convert('L')
    image = image_url.resize((28, 28))

    # make a tensor from the image
    tensor_image = transformer(image)
    return tensor_image

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            print("output: ", output)
            test_loss += criterion(output, target).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("accuracy: ", criterion(output, target).item() * 100)
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def get_prediction(model, device, test_loader, model_name):
    model.eval()
    for data in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        max, index = torch.max(output, dim=1)
        
        print(model_name, " maximum prediction tensor: ", max.item())
        print(model_name, " index + predicted value: ", index.item())
        return index.item()


def main():
    # Training settings
    img = draw_img.start_drawing()
    if img is not None:
        prediction(img)
        main()

def prediction(img):
    args = get_args()
    device = torch.device(args.device)
    
    #image_0 = transform_image(os.path.dirname(__file__) + "\\2.png")
    image_0 = transform_image(img)  
    dataset = [(image_0)]
    
    # load custom images
    custom_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    model = SampleConvNet().to(device)
    model_dp = SampleConvNet().to(device)
    
    checkpoint = torch.load(os.path.dirname(__file__) + '\\mnist_without_dp.pt')
    checkpoint_dp = torch.load(os.path.dirname(__file__) + '\\mnist_with_dp.pt')

    new_checkpoint = {}
    for key in checkpoint.keys():
        new_key = key.replace('_module.', '') # remove '_module.' prefix that's added when you make a model in parallelism (using gpu)
        new_checkpoint[new_key] = checkpoint[key]
    model.load_state_dict(new_checkpoint)

    new_checkpoint_dp = {}
    for key in checkpoint_dp.keys():
        new_key = key.replace('_module.', '') # remove '_module.' prefix that's added when you make a model in parallelism (using gpu)
        new_checkpoint_dp[new_key] = checkpoint_dp[key]
    model_dp.load_state_dict(new_checkpoint_dp)

    model.eval()
    model_dp.eval()

    no_dp_pred = get_prediction(model, device, custom_loader, "No DP - ")
    dp_pred = get_prediction(model_dp, device, custom_loader, "With DP - ")

    window_string = "No DP Prediction: {} \n\nDP Prediction: {}".format(no_dp_pred, dp_pred)
    win32api.MessageBox(0, window_string, "AI Prediction Results")


if __name__ == "__main__":
    win32api.MessageBox(0, "Draw a number on the screen. Once you're done, press ctrl+s to process the image.", "How to use")
    main()

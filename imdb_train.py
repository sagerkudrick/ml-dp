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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
import sys
import importlib
from ml_args import get_args

class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "imdb model"

# ignore
def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc

# used to format the loaded dataset 
def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y

# custom trainer
def train(args, model, train_loader, optimizer, privacy_engine, epoch):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.train().to(device)

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(data).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    if not args.disable_dp:
        print("without accountant")
        epsilon_output = privacy_engine.get_epsilon(delta=args.delta)
        print(epsilon_output)
        print("with accountant")
        epsilon_output = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(epsilon_output)

        print(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {epsilon_output:.2f}, δ = {args.delta})"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} ] \t Accuracy: {np.mean(accuracies):.6f}"
        )

# evaluate 
def evaluate(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.eval().to(device)

    with torch.no_grad():
        for data, label in tqdm(test_loader):

            data = data.to(device)
            label = label.to(device)
            predictions = model(data)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            losses.append(loss.item())
            accuracies.append(acc.item())

    mean_accuracy = np.mean(accuracies)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), mean_accuracy * 100
        )
    )
    return mean_accuracy


def main():
    
    args = get_args()
    device = torch.device(args.device)

    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=args.max_sequence_length
        ),
        batched=True,
    )

    dataset.set_format(type="torch", columns=["input_ids", "label"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # torch.utils.data, gives you an iterable over a dataset 
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    # Model is our custom module that leverages nn.Module, gives access to building neural networks
    model = SampleNet(vocab_size=len(tokenizer)).to(device)

    # pytorch, you choose what algorithm you want to use that updates weights of the model,
    # minimizing the loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            
            # our custom model, extends nn.Module that allows creating neural networks
            module=model,
            
            # torch.optim - choose what algorithm provided by torch that updates weights of a model,
            #minimizing the loss function
            optimizer=optimizer,

            # torch.utils.data, converts dataset into something iterable 
            data_loader=train_loader,
            
            # number of training steps? no documentation
            epochs=args.epochs,

            # target epsilon, smaller is better
            target_epsilon=5,

            # target delta, chance that the datasets released
            target_delta=0.001,

            # maximum norm of per-sample gradients, values over this are clipped to the specified value
            max_grad_norm=args.max_per_sample_grad_norm,

            # standard sampling, required for DP guarantees - unstable when batch-size is 1
            poisson_sampling=False,

            # batch_first = True # set to true by previous, tensor is shape [K, batch_size, ...], if false: [batch_size, ...]

            # loss_reduction = sum/mean # used for aggregating the gradients
        )

    mean_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, privacy_engine, epoch)
        mean_accuracy = evaluate(args, model, test_loader)

    torch.save(model.state_dict(), "imdb_without_dp_test.pt")


if __name__ == "__main__":
    main()

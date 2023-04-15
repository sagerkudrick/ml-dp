import argparse
import os
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
from datasets import Dataset
import datasets
from ml_args import get_args
import win32api

# imdb nn model 
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

# padding elements
def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y

# helper method for returning positive/negative strings
def sentiment_output(prediction):
    if(prediction == 0):
        return "negative review"
    else:
        return "positive review"

def main():

    # get args from ml_args.py; originally was inside each individual .py file
    args = get_args()
    device = torch.device(args.device) # cpu or cuda, default is cuda- need to specify if you want to run on cpu via --device=cpu
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased") # make tensors from input
    model = SampleNet(vocab_size=len(tokenizer)).to(device) # model with the vocabulary size, aka all the words that could be used
    model_dp = SampleNet(vocab_size=len(tokenizer)).to(device) # same as above, but for dp eval

    loaded_dp = torch.load(os.path.dirname(__file__) + '\\models\\imdb_with_dp.pt') # load model with dp
    loaded = torch.load(os.path.dirname(__file__) + '\\models\\imdb_without_dp.pt') # load model WITHOUT dp 

    # make the model for dp
    new_checkpoint_dp = {}
    for key in loaded_dp.keys():
        new_key = key.replace('_module.', '') # remove '_module.' prefix that's added when you make a model in parallelism (using gpu)
        new_checkpoint_dp[new_key] = loaded_dp[key]

    model_dp.load_state_dict(new_checkpoint_dp)
    model_dp.eval()

    # make the model for without dp 
    new_checkpoint = {}
    for key in loaded.keys():
        new_key = key.replace('_module.', '') # remove '_module.' prefix that's added when you make a model in parallelism (using gpu)
        new_checkpoint[new_key] = loaded[key]

    model.load_state_dict(new_checkpoint)
    model.eval()

    # my stuff
    raw_dataset = load_dataset('json', data_files={"train": os.path.dirname(__file__) + '\\imdb_samples\\reviews.json'}, cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    print(raw_dataset["train"][0])
    dataset2 = raw_dataset["train"].map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=args.max_sequence_length
        ),
        batched=True,
    )

    dataset2.set_format(type="torch", columns=["input_ids", "label"])

    trainer = dataset2

    # dataset I made
    train_loader = torch.utils.data.DataLoader(
        trainer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    def binary_accuracy(preds, y):
        correct = (y.long() == torch.argmax(preds, dim=1)).float()
        acc = correct.sum() / len(correct)
        return acc

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
                predictions = model(data).squeeze(1)
                print(predictions)
                print("prediction: ", predictions.argmax(dim=1).float().mean())
                
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
    
    def prediction(args, model, test_loader):
        device = torch.device(args.device)
        model = model.eval().to(device)
        for data, label in tqdm(test_loader):
            data = data.to(device).squeeze(1)
            output = model(data)
            return output
            

    evaluate(args, model, train_loader)
    predictions = prediction(args, model, train_loader) # Tensor([[ 39.6772,  22.7634],[ -5.3484,   7.7618],[  2.2332,  13.6540],[ 21.5630,  17.0304],])
    tensorList = torch.Tensor(predictions)
    
    modal_list = []
    padding = "         "
    for i, tensor in enumerate(tensorList):  
        max, index = torch.max(tensor, dim=0)
        modal_list.append("                           '" + raw_dataset["train"][i]["text"] + "'\nRegular Prediction: " + padding + sentiment_output(index.item()) + "\n")
        print(raw_dataset["train"][i])
        print("maximum prediction rate: ", max.item())
        print("predicted: ", sentiment_output(index.item()), "\n")
    #evaluate(args, model, test_loader)

    evaluate(args, model_dp, train_loader)
    predictions_dp = prediction(args, model_dp, train_loader) # Tensor([[ 39.6772,  22.7634],[ -5.3484,   7.7618],[  2.2332,  13.6540],[ 21.5630,  17.0304],])
    tensorList_dp = torch.Tensor(predictions_dp)
    
    for i, tensor in enumerate(tensorList_dp):  
        max, index = torch.max(tensor, dim=0)
        modal_list[i] += "DP Prediction:          " + padding + sentiment_output(index.item()) + "\n\n"
        print(raw_dataset["train"][i])
        print("dp maximum prediction rate: ", max.item())
        print("dp predicted: ", sentiment_output(index.item()), "\n")
    
    mb_message = ""
    for item in modal_list:
        mb_message += item
    
    win32api.MessageBox(0, mb_message, "AI Prediction Results")


if __name__ == "__main__":
    main()

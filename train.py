import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import time
import wandb
import sys

from model import DvectorModel
from train_dataset import TrainDataset
from test import get_eer

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else print('No GPU'))
    print(device)
    
    train_data_path = '/data/VoxCeleb1/train'
    test_data_path = '/data/VoxCeleb1/test'
    classes = 1211
    learning_rate = 0.001
    embedding_size = 256
    try:
        start_epoch = int(sys.argv[1])
        model = torch.load('model_epoch'+str(start_epoch)+'.pth').to(device)
    except:
        start_epoch = 0
        model = DvectorModel(embedding_size=embedding_size, class_size=classes).to(device)
    epochs = 100
    batch_size = 1024
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.system('wandb login be65d6ddace6bf4e2441a82af03c144eb85bbe65')
    wandb.init(project='dvector-original', entity='dvector')
    wandb.config = {
        "learning_rate" : learning_rate,
        "epochs" : epochs,
        "batch_size" : batch_size
    }

    wandb.define_metric("loss")
    wandb.define_metric("eer")

    train_data = TrainDataset(train_data_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    for epoch in range(epochs):
        model.train()
        for (X, y) in tqdm(train_dataloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            _ , pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            wandb.log({"loss":loss})
        checkpoint_path = 'model_epoch'+str(start_epoch+epoch+1)+'.pth'
        torch.save(model,checkpoint_path)
        eer = get_eer(checkpoint_path=checkpoint_path)
        wandb.log({"eer": eer})



if __name__ == '__main__':
    main()
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
from test_dataset import TestDataset
from test import get_eer

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else print('No GPU'))
    print(device)
    
    # data paths
    train_data_path = '/data/VoxCeleb1/train'
    test_data_path = '/data/VoxCeleb1/test'
    trial_path = '/data/VoxCeleb1/trials/trials.txt'

    # conditions and hyperparameters
    classes = 1211
    learning_rate = 0.001
    embedding_size = 64
    n_mels = 40
    epochs = 300
    batch_size = 512
    loss_fn = nn.CrossEntropyLoss().to(device)
    try:
        start_epoch = int(sys.argv[1])
        model = torch.load('model_epoch'+str(start_epoch)+'.pth').to(device)
    except:
        start_epoch = 0
        model = DvectorModel(
            embedding_size=embedding_size, 
            class_size=classes,
            n_mels=n_mels,
            ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # wandb 설정
    os.system('wandb login be65d6ddace6bf4e2441a82af03c144eb85bbe65')
    wandb.init(project='dvector-original-s2v5', entity='dvector')
    wandb.config = {
        "learning_rate" : learning_rate,
        "epochs" : epochs,
        "batch_size" : batch_size
    }
    wandb.define_metric("loss")
    wandb.define_metric("eer")

    # speaker_id to label 딕셔너리를 만듦.
    speaker_ids = list(map(lambda x:int(x.split('id1')[1]), os.listdir(train_data_path)))
    speaker_ids.sort()
    labels = [i for i in range(len(speaker_ids))]
    speaker_ids_to_labels = {speaker_ids[i]:labels[i] for i in range(len(speaker_ids))}
    print(f'speakers: {len(speaker_ids)}')

    train_data = TrainDataset(train_data_path,speaker_ids_to_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data = TestDataset(test_data_path)

    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
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
        if epoch % 10 == 0:
            scheduler.step()
        test_data.update_embeddings(model,embedding_size,device)
        eer, threshold = get_eer(
            test_dataset=test_data,
            test_path=test_data_path,
            trial_path=trial_path
            )
        print('Threshold: ' + str(threshold))
        print('EER: ' + str(eer))
        wandb.log({"eer": eer})
        if epoch % 100 == 0:
            checkpoint_path = 'model_epoch'+str(start_epoch+epoch+1)+'.pth'
            torch.save(model,checkpoint_path)
    checkpoint_path = 'model_epoch'+str(start_epoch+epoch+1)+'.pth'
    torch.save(model,checkpoint_path)



if __name__ == '__main__':
    main()
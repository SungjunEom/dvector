import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torch import dot
from torch.linalg import norm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd

from model import DvectorModel
from test_dataset import TestDataset

def get_eer(checkpoint_path='model_epoch7.pth'):
    trial_path = '/data/VoxCeleb1/trials/trials.txt'
    test_path = '/data/VoxCeleb1/test'
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = torch.load(checkpoint_path)
    model.to(device)
    model.eval()
    test_dataset = TestDataset(model=model,data_path=test_path,device=device)
    trial_data_list = pd.read_csv(trial_path,names=['positive','file1','file2'],sep=' ')
    labels = trial_data_list.positive
    file1 = list(trial_data_list.file1)
    file2 = list(trial_data_list.file2)
    file1_embeddings = list(map(lambda x: test_dataset.get_embedding(os.path.join(test_path,x)),\
                            file1))
    file2_embeddings = list(map(lambda x: test_dataset.get_embedding(os.path.join(test_path,x)),\
                            file2))
    cos_sims = [cosine_similarity(file1_embeddings[i][0], file2_embeddings[i][0]) for i in range(len(file1))]
    fpr, tpr, _ = metrics.roc_curve(labels, cos_sims, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print('EER:'+str(eer))

def cosine_similarity(a, b):
    return dot(a,b) / (norm(a)*norm(b))

def main():
    get_eer()

if __name__ == '__main__':
    main()
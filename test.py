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

def score():
    pass

def enroll(checkpoint_path,audio_file):
    if wav.shape[0] >= 16000*4:
        start = random.randrange(0,wav.shape[0] - frames + 1)
        wav = wav[start:start+frames]
    else:
        # start = random.randrange(0, frames - wav.shape[0] + 1)
        # wav = np.append(wav,wav[:start+frames])
        wav = np.append(wav,np.zeros(frames - wav.shape[0]))
    device = torch.device('cpu')
    model = torch.load(checkpoint_path)
    model.to(device)
    model.eval()
    x = torch.FloatTensor(wav).to(device)
    x = torch.unsqueeze(x,0)
    x, _ = model(x)
    return x


def get_eer(test_dataset,test_path,trial_path):
    
    trial_data_list = pd.read_csv(trial_path,names=['positive','file1','file2'],sep=' ')
    labels = trial_data_list.positive
    file1 = list(trial_data_list.file1)
    file2 = list(trial_data_list.file2)

    file1_embeddings = list(map(lambda x: test_dataset.get_embedding(os.path.join(test_path,x)),\
                            file1))
    file2_embeddings = list(map(lambda x: test_dataset.get_embedding(os.path.join(test_path,x)),\
                            file2))

    cos_sims = [cosine_similarity(file1_embeddings[i][0], file2_embeddings[i][0]) for i in range(len(file1))]
    fpr, tpr, thresholds = metrics.roc_curve(labels, cos_sims, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr,thresholds)(eer)

    return eer, threshold

def cosine_similarity(a, b):
    return dot(a,b) / (norm(a)*norm(b))

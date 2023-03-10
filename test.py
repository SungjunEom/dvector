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
import sys

from model import DvectorModel
from test_dataset import TestDataset


def get_eer(test_dataset,test_path,trial_path,device='cuda:1'):
    
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

# 두 화자의 코사인 유사도 반환.
# 안씀
def score(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)

# 오디오 파일의 임베딩 구함.
# 안씀
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

if __name__ == '__main__':
    train_data_path = '/data/train'
    test_data_path = '/data/test'
    trial_path = '/data/trials/trials.txt'
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else print('No GPU'))
    checkpoint_path = sys.argv[1]
    embedding_size = int(sys.argv[2])
    model = torch.load(checkpoint_path).to(device)
    test_data = TestDataset(test_data_path)
    test_data.update_embeddings(model,embedding_size,device)

    eer, threshold = get_eer(test_data,test_data_path,trial_path)
    print('EER: '+str(eer))
    print('Threshold: '+str(threshold))
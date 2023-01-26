import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
import random

class TestDataset():
    def __init__(self, model, data_path,device):
        self.root_dir = data_path
        self.speaker_ids = list(map(lambda x:int(x.split('id1')[1]), os.listdir(data_path)))
        self.speaker_wav_paths = {}
        self.append_files_in_dirs(data_path)
        model.eval()
        with torch.no_grad():
            for wav_path in self.speaker_wav_paths:
                wav, sr = librosa.load(wav_path,sr=None)
                frames = 16000*4
                if wav.shape[0] >= 16000*4:
                    start = random.randrange(0,wav.shape[0] - frames + 1)
                    wav = wav[start:start+frames]
                else:
                    # start = random.randrange(0, frames - wav.shape[0] + 1)
                    # wav = np.append(wav,wav[:start+frames])
                    wav = np.append(wav,np.zeros(frames - wav.shape[0]))
                x = torch.FloatTensor(wav).to(device)
                x = torch.unsqueeze(x,0)
                self.speaker_wav_paths[wav_path], _ = model(x)

    def __len__(self):
        len(self.speaker_wav_paths)
        return

    def get_embedding(self,file_path):
        return self.speaker_wav_paths[os.path.join(self.root_dir,file_path)]

    def append_files_in_dirs(self,root_dir):
        dirs = os.listdir(root_dir)
        for id in dirs:
            path = os.path.join(root_dir,id)
            wavs_dir = os.listdir(path)
            for wav_dir in wavs_dir:
                wav_path = os.path.join(path,wav_dir)
                wavs = os.listdir(wav_path)
                for wav in wavs:
                    full_path = os.path.join(wav_path, wav)
                    self.speaker_wav_paths[full_path] = None

        
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
import soundfile as sf
import random

class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.speaker_ids = list(map(lambda x:int(x.split('id1')[1]), os.listdir(data_path)))
        self.speaker_ids.sort()
        self.labels = [i for i in range(len(self.speaker_ids))]
        self.speaker_ids_to_labels = {self.speaker_ids[i]:self.labels[i] for i in range(len(self.speaker_ids))}
        print(f'speakers: {len(self.speaker_ids)}')
        self.speaker_wav_paths = []
        for id in self.speaker_ids:
            speaker_path = data_path+'/id1'+str(id).zfill(4)
            temp_dirs = os.listdir(speaker_path)
            for temp_dir in temp_dirs:
                wavs_path = os.listdir(speaker_path+'/'+temp_dir)
                wavs_path = list(map(lambda x:speaker_path+'/'+temp_dir+'/'+x,wavs_path))
                for wav_path in wavs_path:
                    self.speaker_wav_paths.append((id,wav_path))

    def __len__(self):
        return len(self.speaker_wav_paths)

    def __getitem__(self,idx):
        speaker_id, wav_path = self.speaker_wav_paths[idx]
        wav, sr = sf.read(wav_path)
        # wav, sr = librosa.load(wav_path, sr=None)
        
        # 아래 지워도 됨
        # S = librosa.feature.melspectrogram(y=wav,sr=sr,n_mels=13,fmax=sr/2)
        # if S.shape[1] >= 237: # mel spectrogram에서 x축 237개 포인트로 제한
        #     S = S[:,:237]
        # else:
        #     S = np.c_[S,S[:,:237-S.shape[1]]]


        # 4초만 가져오기
        frames = 16000*4
        if wav.shape[0] >= 16000*4:
            start = random.randrange(0,wav.shape[0] - frames + 1)
            wav = wav[start:start+frames]
        else:
            # start = random.randrange(0, frames - wav.shape[0] + 1)
            # wav = np.append(wav,wav[:start+frames])
            wav = np.append(wav,np.zeros(frames - wav.shape[0]))
        return torch.FloatTensor(wav), self.speaker_ids_to_labels[speaker_id]
        
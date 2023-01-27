import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
import soundfile as sf
import random

class TrainDataset(Dataset):
    def __init__(self, data_path):
        # speaker_id 리스트를 만듦.
        self.speaker_ids = list(map(lambda x:int(x.split('id1')[1]), os.listdir(data_path)))
        self.speaker_ids.sort()

        # speaker_id가 중간에 빠져있는 값들이 있기 때문에
        # CrossEntropy계산을 위해 라벨을 붙여주고
        # 딕셔너리로 대응시킴.
        self.labels = [i for i in range(len(self.speaker_ids))]
        self.speaker_ids_to_labels = {self.speaker_ids[i]:self.labels[i] for i in range(len(self.speaker_ids))}
        print(f'speakers: {len(self.speaker_ids)}')

        # speaker_id에 대응되는 wav파일들을 튜플로 짝짓고
        # 튜플들의 리스트를 만듦.
        # ex) [(id, 파일경로),...]
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

        # 4초만 가져오기
        frames = 16000*4
        if wav.shape[0] >= frames:
            start = random.randrange(0,wav.shape[0] - frames + 1)
            wav = wav[start:start+frames]
        else:
            wav = np.append(wav,np.zeros(frames - wav.shape[0]))
        return torch.FloatTensor(wav), self.speaker_ids_to_labels[speaker_id]
        
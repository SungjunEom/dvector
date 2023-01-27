import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
import soundfile as sf
import random

class TestDataset():
    def __init__(self, model, data_path,device):
        self.root_dir = data_path

        # test데이터셋에 있는 wav파일 목록을 가져와서
        # 딕셔너리로 만듦.
        # ex) speaker_wav_paths = {'/data/VoCeleb1/test/id10295/3tvnlmkCiTw/00008.wav':None, ...}
        self.speaker_wav_paths = {}
        dirs = os.listdir(self.root_dir)
        for id in dirs:
            path = os.path.join(self.root_dir,id)
            wavs_dir = os.listdir(path)
            for wav_dir in wavs_dir:
                wav_path = os.path.join(path,wav_dir)
                wavs = os.listdir(wav_path)
                for wav in wavs:
                    full_path = os.path.join(wav_path, wav)
                    self.speaker_wav_paths[full_path] = None
        
        # test데이터셋에 있는 wav를 4초 자른 뒤 모델에 통과시키고
        # 나온 임베딩을 speaker_wav_paths의 값으로 함.
        model.eval()
        with torch.no_grad():
            for wav_path in self.speaker_wav_paths:
                wav, sr = sf.read(wav_path)
                frames = 16000*4
                if wav.shape[0] >= frames:
                    start = random.randrange(0,wav.shape[0] - frames + 1)
                    wav = wav[start:start+frames]
                else:
                    wav = np.append(wav,np.zeros(frames - wav.shape[0]))
                x = torch.FloatTensor(wav).to(device)
                x = torch.unsqueeze(x,0)
                x, _ = model(x)
                self.speaker_wav_paths[wav_path] = x.cpu()

        print(self.speaker_wav_paths['/data/VoxCeleb1/test/id10290/O-V_sInAw5M/00007.wav'])

    def __len__(self):
        len(self.speaker_wav_paths)
        return

    # wav파일 경로를 받으면 미리 저장되어있는 임베딩을 출력함.
    def get_embedding(self,file_path):
        return self.speaker_wav_paths[file_path]
        

        
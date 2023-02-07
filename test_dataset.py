import torch
from torch.utils.data import Dataset
import os
import numpy as np
import librosa
import soundfile as sf
import random

class TestDataset():
    def __init__(self, data_path):
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

    def update_embeddings(self,model,embedding_size,device):
        
        # test데이터셋에 있는 wav를 4초 잘라서 5개의 구간을 만들고
        # 5개의 구간에 대한 평균 임베딩을 구함
        # (wav크기가 20이상이 대부분이므로)
        sample_num = 5
        model.eval()
        with torch.no_grad():
            for wav_path in self.speaker_wav_paths:
                wav, sr = sf.read(wav_path)
                frames = 16000*4
                temp_embedding = torch.zeros(1,embedding_size)
                fragments = np.linspace(0, wav.shape[0]-frames,5)
                fragments = fragments.astype(np.int64)
                x = []
                for fragment in fragments:
                    if wav.shape[0] >= frames:
                        wav2 = wav[fragment:fragment+frames]
                    else:
                        wav2 = np.append(wav,np.zeros(frames - wav.shape[0]))
                    x.append(torch.FloatTensor(wav2))
                    # x = torch.FloatTensor(wav2).to(device)
                    # x = torch.unsqueeze(x,0)
                    # x, _ = model(x)
                    # temp_embedding += x.cpu()
                x = torch.stack(x,dim=0)
                x = x.to(device)
                x, _ = model(x)
                x = torch.sum(x,dim=0) / sample_num
                self.speaker_wav_paths[wav_path] = torch.unsqueeze(x.cpu(), 0)
    def __len__(self):
        return len(self.speaker_wav_paths)
        

    # wav파일 경로를 받으면 미리 저장되어있는 임베딩을 출력함.
    def get_embedding(self,file_path):
        return self.speaker_wav_paths[file_path]
        

        
from torch import nn
import torch
import torchaudio

class DvectorModel(nn.Module):
    def __init__(self, embedding_size=1024,class_size=1211,n_mels=13):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(         
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, 
                n_fft=512, 
                win_length=400, 
                hop_length=160, 
                f_min = 20, 
                f_max = 7600, 
                window_fn=torch.hamming_window, 
                n_mels=n_mels
                ),
            )
        
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(n_mels*401,embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.linear3 = nn.Linear(embedding_size, embedding_size)
        self.linear4 = nn.Linear(embedding_size, class_size)
        self.batchnorm1 = nn.BatchNorm1d(embedding_size)
        self.batchnorm2 = nn.BatchNorm1d(embedding_size)
        self.batchnorm3 = nn.BatchNorm1d(embedding_size)
        self.batchnorm4 = nn.BatchNorm1d(class_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
    
    def forward(self,x):
        x = self.torchfbank(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.batchnorm1(self.linear1(x)))
        x = self.dropout2(self.activation(self.batchnorm2(self.linear2(x))))
        speaker_embedding = self.linear3(x)
        x = self.dropout3(self.activation(self.batchnorm3(speaker_embedding)))
        x = self.batchnorm4(self.linear4(x))
        return speaker_embedding, x
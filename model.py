from torch import nn
import torch
import torchaudio

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return x*self.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1,1,3)
        self.batchnorm1 = nn.BatchNorm2d

class CNN(nn.Module):
    def __init__(
        self, 
        embedding_size=1024,
        class_size=1211,
        n_mels=13,
        ):
        super().__init__()
        self.torchfbank = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, 
                n_fft=512, 
                win_length=512, 
                hop_length=160, 
                f_min = 20, 
                f_max = 8000, 
                window_fn=torch.hamming_window, 
                n_mels=n_mels
                )

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(3,3,3)
        self.conv3 = nn.Conv2d(3,3,7,stride=2)
        self.conv4 = nn.Conv2d(3,3,7,stride=2)
        self.linear5 = nn.Linear(1425,embedding_size)
        self.linear6 = nn.Linear(embedding_size,class_size)

    def forward(self,x):
        x = self.torchfbank(x)
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        speaker_embedding = self.linear5(x)
        x = self.linear6(speaker_embedding)
        return speaker_embedding, x

class DvectorModel(nn.Module):
    def __init__(
        self, 
        embedding_size=1024,
        class_size=1211,
        n_mels=13,
        ):
        super().__init__()
        self.torchfbank = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, 
                n_fft=512, 
                win_length=512, 
                hop_length=160, 
                f_min = 20, 
                f_max = 8000, 
                window_fn=torch.hamming_window, 
                n_mels=n_mels
                )
        
        self.activation = Swish()
        self.maxpool1d = nn.MaxPool1d(2)
        self.linear1 = nn.Linear(n_mels*401,embedding_size*16)
        self.linear2 = nn.Linear(embedding_size*8, embedding_size*8)
        self.linear3 = nn.Linear(embedding_size*4, embedding_size*4)
        self.linear4 = nn.Linear(embedding_size*2, embedding_size*2)
        self.linear5 = nn.Linear(embedding_size, class_size)
        self.batchnorm1 = nn.BatchNorm1d(embedding_size*16)
        self.batchnorm2 = nn.BatchNorm1d(embedding_size*8)
        self.batchnorm3 = nn.BatchNorm1d(embedding_size*4)
        self.batchnorm4 = nn.BatchNorm1d(embedding_size*2)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)
    
    def forward(self,x):
        x = self.torchfbank(x)
        x = torch.flatten(x, start_dim=1)
        x = self.maxpool1d(self.activation(self.batchnorm1(self.linear1(x))).unsqueeze(1)).squeeze(1)
        x = self.maxpool1d(self.activation(self.batchnorm2(self.linear2(x))).unsqueeze(1)).squeeze(1)
        x = self.maxpool1d(self.dropout3(self.activation(self.batchnorm3(self.linear3(x)))).unsqueeze(1)).squeeze(1)
        speaker_embedding = self.maxpool1d(self.dropout4(self.activation(self.batchnorm4(self.linear4(x)))).unsqueeze(1)).squeeze(1)
        x = self.linear5(speaker_embedding)
        return speaker_embedding, x
from torch import nn
import torch
import torchaudio

class DvectorModel(nn.Module):
    def __init__(self, embedding_size=1024,class_size=1211):
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
                n_mels=13
                ),
            )
        
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(13*401,embedding_size)
        self.batchnorm1 = nn.BatchNorm1d(embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.batchnorm2 = nn.BatchNorm1d(embedding_size)
        self.linear3 = nn.Linear(embedding_size, embedding_size)
        self.batchnorm3 = nn.BatchNorm1d(embedding_size)
        self.linear4 = nn.Linear(embedding_size, class_size)
        self.batchnorm4 = nn.BatchNorm1d(class_size)
    
    def forward(self,x):
        x = self.torchfbank(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.batchnorm1(self.linear1(x)))
        x = self.activation(self.batchnorm2(self.linear2(x)))
        speaker_embedding = self.activation(self.batchnorm3(self.linear3(x)))
        x = self.batchnorm4(self.linear4(speaker_embedding))
        return speaker_embedding, x
    
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = DvectorModel().to(device)
    summary(model, input_size=(13*401,), device=device)
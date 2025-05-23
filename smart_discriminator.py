import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, n_channels=64, seq_len=120000, do=.1):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, n_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
            
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1),
           # nn.BatchNorm1d(n_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
            
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2, padding=1),
           # nn.BatchNorm1d(n_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
            
            nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=4, stride=2, padding=1),
           # nn.BatchNorm1d(n_channels * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
            
            nn.Conv1d(n_channels * 8, n_channels * 16, kernel_size=4, stride=2, padding=1),
           # nn.BatchNorm1d(n_channels * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
        )
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(n_channels * 16, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(do),
            nn.Linear(128, 1)  # NO sigmoid here!
        )

    def forward(self, x):
        x = self.model(x)
        x = self.global_max_pool(x).squeeze(-1)
        output = self.fc(x)
        return output  # Returns logits, use BCEWithLogitsLoss

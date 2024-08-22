class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c0 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.c1 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, padding=1)

        self.d = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.mlp1 = nn.Linear(24*30, 25)
        self.mlp2 = nn.Linear(25+1, 3)
        
    def forward(self, x, ps):
        x = self.d(self.tanh(self.c0(x)))
        x = self.d(self.tanh(self.c1(x)))
        x = self.d(self.tanh(self.c2(x)))
        return self.tanh(self.mlp2(torch.concatenate([self.d(self.tanh(self.mlp1(self.flatten(x)))), ps], axis=1)))

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c0 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.c1 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)

        self.d = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.mlp1 = nn.Linear(2160, 25)
        self.mlp2 = nn.Linear(25+1, 3)
        
    def forward(self, x, ps):
        x = x.unsqueeze(1)
        x = self.d(self.tanh(self.c0(x)))
        x = self.d(self.tanh(self.c1(x)))
        x = self.d(self.tanh(self.c2(x)))
        return self.tanh(self.mlp2(torch.concatenate([self.d(self.tanh(self.mlp1(self.flatten(x)))), ps], axis=1)))

class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c0 = nn.Linear(30, 64)
        self.c1 = nn.Linear(64, 32)
        self.c2 = nn.Linear(32, 16)

        self.d = nn.Dropout(0.3)
        
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.mlp1 = nn.Linear(16*3, 8)
        self.mlp2 = nn.Linear(8+1, 3)
        
    def forward(self, x, ps):
        x = x.unsqueeze(1)
        x = self.d(self.tanh(self.c0(x)))
        x = self.d(self.tanh(self.c1(x)))
        x = self.d(self.tanh(self.c2(x)))
        return self.tanh(self.mlp2(torch.concatenate([self.d(self.tanh(self.mlp1(self.flatten(x)))), ps], axis=1)))

class STFTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c0 = nn.Conv2d(in_channels=33, out_channels=16, kernel_size=3, padding=1).to(device)
        self.c1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1).to(device)
        self.c2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1).to(device)
        self.c3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1).to(device)
        
        self.d = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.mlp1 = nn.Linear(396, 32)
        self.mlp2 = nn.Linear(32+1, 3)
        
    def forward(self, x, ps):
        x = torch.concatenate([torch.view_as_real(torch.stft(x[:, i], 
                                                             n_fft=64, 
                                                             hop_length=2, 
                                                             win_length=64, 
                                                             return_complex=True)) for i in range(3)], 
                                                  axis=-1).to(device),
        x = self.d(self.tanh(self.c0(x)))
        x = self.d(self.tanh(self.c1(x)))
        x = self.d(self.tanh(self.c2(x)))
        x = self.d(self.tanh(self.c3(x)))
        return self.tanh(self.mlp2(torch.concatenate([self.d(self.tanh(self.mlp1(self.flatten(x)))), ps], axis=1)))
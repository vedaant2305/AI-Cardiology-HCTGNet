%%writefile model.py
import torch
import torch.nn as nn

class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

class HCTGNet(nn.Module):
    def __init__(self, num_classes=5):
        super(HCTGNet, self).__init__()
        self.cnn_branch = CNNBranch()
        
        # Transformer Branch (Fixed d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # Global Pooling and Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 188]
        cnn_features = self.cnn_branch(x) # [batch, 64, 47]
        
        # Prepare for Transformer: [batch, seq_len, d_model]
        x_trans = cnn_features.permute(0, 2, 1) 
        trans_out = self.transformer_encoder(x_trans)
        
        # Back to [batch, d_model, seq_len] for pooling
        out = trans_out.permute(0, 2, 1)
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out)
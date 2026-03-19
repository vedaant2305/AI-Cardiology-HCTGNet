import torch
import torch.nn as nn
import math

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.skip_projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip_projection = nn.Identity()
            
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.skip_projection(x)
        out = self.conv_path(x)
        return self.relu(out + shortcut)

class CNNBranch(nn.Module):
    def __init__(self, cnn_dims=[32, 64, 128]):
        super(CNNBranch, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, cnn_dims[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(cnn_dims[0]),
            nn.ReLU()
        )
        self.res_block1 = ResBlock(cnn_dims[0], cnn_dims[0], stride=1)
        self.res_block2 = ResBlock(cnn_dims[0], cnn_dims[1], stride=2)
        self.res_block3 = ResBlock(cnn_dims[1], cnn_dims[2], stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.global_pool(x).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBranch(nn.Module):
    def __init__(self, d_model=64, num_layers=2):
        super(TransformerBranch, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Transpose for embedding: [batch, 1, 188] -> [batch, 188, 1]
        x = x.permute(0, 2, 1) 
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        # Transpose back for pooling: [batch, 188, d_model] -> [batch, d_model, 188]
        x = x.permute(0, 2, 1) 
        return self.global_pool(x).squeeze(-1)

class GatedFusion(nn.Module):
    def __init__(self, cnn_dim=128, trans_dim=64, out_dim=128):
        super(GatedFusion, self).__init__()
        self.proj_cnn = nn.Linear(cnn_dim, out_dim)
        self.proj_trans = nn.Linear(trans_dim, out_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(cnn_dim + trans_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, trans_feat):
        cnn_proj = self.proj_cnn(cnn_feat)
        trans_proj = self.proj_trans(trans_feat)
        gate = self.gate_mlp(torch.cat([cnn_feat, trans_feat], dim=1))
        return gate * cnn_proj + (1 - gate) * trans_proj

class Classifier(nn.Module):
    def __init__(self, in_dim=128, num_classes=5):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.head(x)

class HCTGNet(nn.Module):
    def __init__(self, num_classes=5, cnn_dims=[32, 64, 128], d_model=64, fusion_dim=128):
        super(HCTGNet, self).__init__()
        self.cnn_branch = CNNBranch(cnn_dims)
        self.transformer_branch = TransformerBranch(d_model)
        self.fusion = GatedFusion(cnn_dims[-1], d_model, fusion_dim)
        self.classifier = Classifier(fusion_dim, num_classes)

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)
        trans_feat = self.transformer_branch(x)
        fused_feat = self.fusion(cnn_feat, trans_feat)
        return self.classifier(fused_feat)

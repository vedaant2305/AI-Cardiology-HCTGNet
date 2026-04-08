import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from model import HCTGNet
from ptbxl_pipeline import PTBXLConfig, PTBXLDataset, load_ptbxl_database, z_score_normalise, split_dataset, get_device

cfg    = PTBXLConfig()
device = get_device()

print("Loading PTB-XL data...")
segments, labels = load_ptbxl_database(cfg)
segments = z_score_normalise(segments)
splits   = split_dataset(segments, labels, cfg)

print("Loading saved checkpoint...")
ckpt  = torch.load('./results/ptbxl/best_ptbxl_model.pth',
                   map_location=device, weights_only=False)
model = HCTGNet(num_classes=5).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

test_loader = DataLoader(
    PTBXLDataset(splits['test']['X'], splits['test']['y']),
    batch_size=512, shuffle=False, num_workers=0,
)

all_preds, all_targets, all_probs = [], [], []
with torch.no_grad():
    for signals, lbls in test_loader:
        logits = model(signals.to(device))
        probs  = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_targets.extend(lbls.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs   = np.array(all_probs)

unique_classes     = sorted(np.unique(all_targets).tolist())
y_bin              = label_binarize(all_targets, classes=unique_classes)
all_probs_filtered = all_probs[:, unique_classes]
auc = roc_auc_score(y_bin, all_probs_filtered,
                    multi_class='ovr', average='macro')

print(f"\n========================================")
print(f"  PTB-XL FINAL RESULTS")
print(f"========================================")
print(f"  Accuracy  : {(all_preds == all_targets).mean():.4f}")
print(f"  Macro AUC : {auc:.4f}")
print(f"========================================")
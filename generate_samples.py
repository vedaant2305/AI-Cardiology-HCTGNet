import os
import numpy as np
from preprocessing import build_mitbih_pipeline, CLASS_NAMES

# Force save to the same folder as this script
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Saving files to: {SAVE_DIR}")

print("Loading test data...")
loaders = build_mitbih_pipeline(batch_size=256, num_workers=0)
test_loader = loaders['test']

saved = {}

for signals, labels in test_loader:
    for i in range(len(labels)):
        cls = labels[i].item()
        if cls not in saved:
            beat = signals[i].squeeze().numpy()
            filename = os.path.join(SAVE_DIR, f"sample_class_{CLASS_NAMES[cls]}.csv")
            np.savetxt(filename, beat, delimiter=",")
            saved[cls] = filename
            print(f"✅ Saved {filename}")
        if len(saved) == 5:
            break
    if len(saved) == 5:
        break

print("\n✅ All 5 sample files created!")
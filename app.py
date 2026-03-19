
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import HCTGNet

# Page Config
st.set_page_config(page_title="AI Cardiology Dashboard", page_icon="🫀")

@st.cache_resource
def load_model():
    model = HCTGNet(num_classes=5)
    
    # Automatically switch between GPU and CPU for local PC use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the weights
    state_dict = torch.load('best_hctg_net.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, device

st.title("🫀 HCTG-Net Arrhythmia Classifier")
st.write("Upload an ECG heartbeat CSV (188 samples) for real-time AI diagnosis.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    heartbeat = data.values.flatten()
    
    if len(heartbeat) == 188:
        # Plotting
        fig, ax = plt.subplots()
        ax.plot(heartbeat, color='#ff4b4b')
        ax.set_title("ECG Morphological Trace")
        st.pyplot(fig)
        
        # Prediction
        model, device = load_model()
        input_tensor = torch.tensor(heartbeat, dtype=torch.float32).view(1, 1, 188).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)
            
        classes = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
        
        st.subheader(f"Diagnosis: {classes[pred.item()]}")
        st.write(f"Confidence Score: {confidence.item()*100:.2f}%")
    else:
        st.error(f"Invalid file format. Expected 188 samples, got {len(heartbeat)}.")

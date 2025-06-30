import streamlit as st
from scapy.all import rdpcap
import numpy as np
from tensorflow.keras.models import load_model

# --- Load model ---
model = load_model("Full_Model_model.h5")

# --- Packet Feature Extractor ---
def extract_features(pkt):
    length = len(pkt)
    proto = pkt.proto if hasattr(pkt, 'proto') else 0
    return [length, proto] + [0] * 48  # pad to 50 features

def classify_packet(pkt):
    features = extract_features(pkt)
    features = np.array(features).reshape(1, 50, 1).astype('float32')
    pred = model.predict(features)
    return 'Attack' if pred[0][0] > 0.5 else 'Normal'

# --- Streamlit UI ---
st.title("🔐 Real-Time Packet Classifier")
st.markdown("Upload a .pcap file and get prediction results per packet.")

uploaded_file = st.file_uploader("Upload PCAP File", type=["pcap"])

if uploaded_file is not None:
    st.success("PCAP file loaded.")
    packets = rdpcap(uploaded_file)
    results = []

    for i, pkt in enumerate(packets):
        label = classify_packet(pkt)
        results.append((i+1, label))

    st.write("### Classification Results")
    st.table(results)
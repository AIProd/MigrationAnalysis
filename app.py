import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.pyplot as plt
import io

# ---------------- Helper functions ----------------
def compute_open_area(image):
    """Compute open wound area % using Otsu threshold."""
    gray = rgb2gray(np.array(image))
    thr = threshold_otsu(gray)
    open_mask = gray >= thr
    open_area_pct = 100.0 * open_mask.mean()

    # Overlay mask for visualization
    overlay = np.array(image).copy()
    overlay[open_mask] = [255, 0, 0]  # red overlay for open area
    return open_area_pct, overlay

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Wound Healing Analyzer", layout="wide")

st.title("🧪 Wound Healing Analyzer")
st.markdown("Upload images at each **timepoint** for a given **concentration**, "
            "and see open area detection with segmentation overlay.")

# Input concentration
concentration = st.selectbox(
    "Select Concentration (p/mL):",
    ["8000", "16000", "32000", "64000", "Control"]
)

# Define standard timepoints
timepoints = [0, 12, 24, 36]
uploaded = {}

st.subheader(f"Upload Images for {concentration} p/mL")

for t in timepoints:
    file = st.file_uploader(f"Upload image at {t}h", type=["png","jpg","jpeg"], key=f"file_{t}")
    if file:
        img = Image.open(file).convert("RGB")
        open_area_pct, overlay = compute_open_area(img)

        st.markdown(f"**Timepoint {t}h — Open Area: {open_area_pct:.2f}%**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"Original ({t}h)", use_column_width=True)
        with col2:
            st.image(overlay, caption=f"Detected Open Area ({t}h)", use_column_width=True)

        uploaded[t] = open_area_pct

# ---------------- Results summary ----------------
if uploaded:
    df = pd.DataFrame({
        "Hours": list(uploaded.keys()),
        "Open Wound Area (%)": list(uploaded.values())
    }).sort_values("Hours")

    st.subheader("📊 Results Summary")
    st.dataframe(df.set_index("Hours"))

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df["Hours"], df["Open Wound Area (%)"], marker="o", linewidth=2,
            color="#56B4E9", label=f"{concentration} p/mL")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Open Wound Area (%)")
    ax.set_title("Wound Area Over Time")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

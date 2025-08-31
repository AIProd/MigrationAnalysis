import streamlit as st
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- Helper ----------------
def compute_open_area(image):
    """Compute open wound area % using Otsu threshold."""
    gray = rgb2gray(np.array(image))
    thr = threshold_otsu(gray)
    open_mask = gray >= thr
    open_area_pct = 100.0 * open_mask.mean()

    # Overlay mask (red = detected open area)
    overlay = np.array(image).copy()
    overlay[open_mask] = [255, 0, 0]
    return open_area_pct, overlay

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Wound Healing Analyzer", layout="wide")

st.title("🧪 Baseline-Normalized Wound Healing Analyzer")
st.markdown("""
Upload **all 4 images** for a given concentration (0h, 12h, 24h, 36h).  
The system will:
1. Detect open wound area % per image.  
2. Use **0h** as the baseline.  
3. Report relative open % and closure % at each stage.  
""")

# Select concentration
concentration = st.selectbox("Select Concentration (p/mL):", 
                             ["8000", "16000", "32000", "64000", "Control"])

# Upload images
timepoints = [0, 12, 24, 36]
uploads = {}

st.subheader(f"Upload images for {concentration} p/mL")

for t in timepoints:
    file = st.file_uploader(f"Upload image at {t}h", type=["png","jpg","jpeg"], key=f"tp{t}")
    if file:
        img = Image.open(file).convert("RGB")
        open_area_pct, overlay = compute_open_area(img)

        st.markdown(f"**{t}h — Raw Open Area: {open_area_pct:.2f}%**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"Original ({t}h)", use_column_width=True)
        with col2:
            st.image(overlay, caption=f"Detection Overlay ({t}h)", use_column_width=True)

        uploads[t] = open_area_pct

# Process once all timepoints uploaded
if len(uploads) == len(timepoints):
    st.success("✅ All timepoints uploaded. Computing baseline-normalized metrics...")

    baseline = uploads[0]  # 0h reference
    summary = []
    for t in timepoints:
        raw = uploads[t]
        rel_open = (raw / baseline) * 100.0 if baseline > 0 else np.nan
        closure = 100.0 - rel_open
        summary.append({"Hours": t,
                        "Raw Open Area %": raw,
                        "Relative Open %": rel_open,
                        "Closure %": closure})

    df = pd.DataFrame(summary).set_index("Hours")

    st.subheader("📊 Baseline-Normalized Results")
    st.dataframe(df.style.format("{:.2f}"))

    # Plot closure over time
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df.index, df["Closure %"], marker="o", linewidth=2,
            color="#009E73", label=f"{concentration} p/mL")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Closure % (relative to 0h)")
    ax.set_title(f"Closure Curve for {concentration} p/mL")
    ax.set_ylim(0,100)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please upload all 4 timepoint images (0h, 12h, 24h, 36h).")

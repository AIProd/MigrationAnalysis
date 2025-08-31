import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Wound Healing (Baseline-Normalized)", layout="wide")
TIMEPOINTS = [0, 12, 24, 36]
CONCENTRATIONS = ["Control", "8000", "16000", "32000", "64000"]

# --------------------------- UTILITIES ----------------------------
@st.cache_data(show_spinner=False)
def _downscale(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_h, new_w = int(h * scale), int(w * scale)
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

def illumination_correct(gray: np.ndarray, sigma_bg: float) -> np.ndarray:
    """Divide by a heavy gaussian to flatten illumination; rescale to [0,1]."""
    bg = gaussian(gray, sigma=sigma_bg, preserve_range=True)
    corr = gray / np.clip(bg, 1e-6, None)
    corr = exposure.rescale_intensity(corr, in_range='image', out_range=(0, 1))
    return corr

def local_std(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Compute local standard deviation via Gaussian moments."""
    m1 = gaussian(gray, sigma=sigma, preserve_range=True)
    m2 = gaussian(gray * gray, sigma=sigma, preserve_range=True)
    var = np.clip(m2 - m1 * m1, 0, None)
    std = np.sqrt(var)
    std = std / (std.max() + 1e-8)
    return std

def segment_open(gray01: np.ndarray, std_sigma: float, sens: float,
                 open_radius: int, close_radius: int, min_area: int) -> np.ndarray:
    """Texture-based segmentation: open region = low texture std."""
    s = local_std(gray01, sigma=std_sigma)
    thr = threshold_otsu(s)
    thr = thr * (1.0 + sens)  # sens in roughly [-0.3, +0.3]
    mask_open = s <= thr
    if open_radius > 0:
        mask_open = binary_opening(mask_open, footprint=disk(open_radius))
    if close_radius > 0:
        mask_open = binary_closing(mask_open, footprint=disk(close_radius))
    if min_area > 0:
        mask_open = remove_small_objects(mask_open, min_size=min_area)
    return mask_open

def center_roi_mask(h: int, w: int, margin_frac: float) -> np.ndarray:
    """Keep a centered ROI, crop margins on all sides by margin_frac."""
    r0, r1 = int(h * margin_frac), int(h * (1 - margin_frac))
    c0, c1 = int(w * margin_frac), int(w * (1 - margin_frac))
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay red mask on image."""
    out = rgb.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    out = (out * (1 - alpha) + red * alpha * mask[..., None]).astype(np.uint8)
    return out

def analyze_image(pil_image: Image.Image, roi_margin: float, bg_sigma: float,
                  std_sigma: float, sens: float, open_r: int, close_r: int, min_area: int):
    """Return (raw_open_pct, overlay_png_bytes)."""
    rgb = np.array(pil_image.convert("RGB"))
    rgb = _downscale(rgb, max_side=1600)
    gray = rgb2gray(rgb).astype(np.float32)
    # illumination correction
    corr = illumination_correct(gray, sigma_bg=bg_sigma)
    # segmentation
    mask_open = segment_open(corr, std_sigma=std_sigma, sens=sens,
                             open_radius=open_r, close_radius=close_r, min_area=min_area)
    # ROI
    h, w = mask_open.shape
    roi = center_roi_mask(h, w, roi_margin)
    valid = mask_open & roi
    raw_open_pct = 100.0 * valid.mean()

    # nice overlay (show ROI outline)
    overlay = overlay_mask((corr * 255).astype(np.uint8)[..., None].repeat(3, axis=2),
                           valid, alpha=0.45)
    # draw ROI rectangle
    rr0, rr1 = int(h * roi_margin), int(h * (1 - roi_margin))
    cc0, cc1 = int(w * roi_margin), int(w * (1 - roi_margin))
    overlay[rr0:rr1, [cc0, cc1 - 1]] = (0, 255, 0)
    overlay[[rr0, rr1 - 1], cc0:cc1] = (0, 255, 0)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return raw_open_pct, buf.getvalue()

def summarize_series(raw_by_t: dict):
    """Return DataFrame with Raw, Relative Open %, Closure % (baseline 0h)."""
    baseline = raw_by_t.get(0, np.nan)
    rows = []
    for t in sorted(raw_by_t.keys()):
        raw = raw_by_t[t]
        rel_open = (raw / baseline) * 100.0 if baseline and not np.isnan(baseline) else np.nan
        closure = 100.0 - rel_open if rel_open == rel_open else np.nan
        rows.append({"Hours": t, "Raw Open %": raw, "Relative Open %": rel_open, "Closure %": closure})
    df = pd.DataFrame(rows).set_index("Hours")
    return df, baseline

# ------------------------------ UI -------------------------------
# --- controls top row: compact, horizontal ---
c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.1, 1.2, 1.2, 1.2, 1.2])
with c1:
    concentration = st.selectbox("Concentration", CONCENTRATIONS, index=1)
with c2:
    roi_margin = st.slider("ROI margin", 0.05, 0.25, 0.12, 0.01, help="Crop margins to avoid scale bar/edges")
with c3:
    bg_sigma = st.slider("BG sigma", 10.0, 80.0, 40.0, 2.0, help="Illumination correction blur")
with c4:
    std_sigma = st.slider("Texture sigma", 3.0, 30.0, 12.0, 1.0, help="Neighborhood for local std")
with c5:
    sens = st.slider("Sensitivity", -0.35, 0.35, 0.00, 0.01, help="Lower → more open; Higher → less open")
with c6:
    morph = st.select_slider("Cleanup (open/close)", options=[(1,1),(2,2),(3,2),(3,3),(4,3)],
                             value=(2,2), help="(opening radius, closing radius)")
open_r, close_r = morph
min_area = 600  # small object removal (px)

# --- upload widgets (compact grid) ---
st.markdown("#### Upload images at each timepoint")
uc1, uc2, uc3, uc4 = st.columns(4)
uploads = {}
for t, col in zip(TIMEPOINTS, [uc1, uc2, uc3, uc4]):
    with col:
        f = st.file_uploader(f"{t} h", type=["png","jpg","jpeg"], key=f"f{t}", label_visibility="visible")
        if f is not None:
            try:
                img = Image.open(f).convert("RGB")
                uploads[t] = img
                st.image(img, caption=f"{t}h", use_column_width=True)
            except Exception:
                st.error("Could not read image.")

st.divider()
go = st.button("▶️ Analyze", type="primary", use_container_width=True)

# ---------------------------- ANALYSIS ---------------------------
if go and uploads:
    # analyze each timepoint present (does not force all four—still shows partial)
    raw = {}
    overlays = {}
    for t in sorted(uploads.keys()):
        raw[t], overlay_png = analyze_image(
            uploads[t], roi_margin=roi_margin, bg_sigma=bg_sigma,
            std_sigma=std_sigma, sens=sens, open_r=open_r, close_r=close_r, min_area=min_area
        )
        overlays[t] = overlay_png

    # LAYOUT: left = overlays grid; right = table + chart + download
    left, right = st.columns([1.2, 1.0])

    with left:
        st.markdown("#### Detection overlays")
        grid = st.columns(2)
        for i, t in enumerate(sorted(overlays.keys())):
            with grid[i % 2]:
                st.image(overlays[t], caption=f"{t}h — overlay", use_column_width=True)

    with right:
        st.markdown("#### 📊 Baseline-Normalized Results")
        df, baseline = summarize_series(raw)
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
        # Plot closure (dynamic limits to show negative if needed)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        x = df.index.values
        y = df["Closure %"].values.astype(float)
        if np.isfinite(y).any():
            ymin = float(np.nanmin(y))
            ymax = float(np.nanmax(y))
            ylo = min(-5, ymin - 5) if np.isfinite(ymin) else -5
            yhi = max(100, ymax + 5) if np.isfinite(ymax) else 100
        else:
            ylo, yhi = -5, 100
        ax.plot(x, y, marker="o", linewidth=2, color="#009E73",
                label=f"{concentration} p/mL")
        ax.set_xlabel("Hours"); ax.set_ylabel("Closure % (relative to 0h)")
        ax.set_title(f"Closure Curve — {concentration} p/mL (baseline {baseline:.2f}% open)")
        ax.grid(True, linestyle="--", alpha=0.5); ax.set_ylim(ylo, yhi)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        # CSV download
        csv = df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"results_{concentration}.csv", use_container_width=True)

else:
    st.info("Upload at least one image and click **Analyze**. For baseline normalization, include 0h.")

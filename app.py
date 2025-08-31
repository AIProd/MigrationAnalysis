import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Migration Image Analysis", layout="wide")
TIMEPOINTS = [0, 12, 24, 36]
CONCENTRATIONS = ["Control", "8000", "16000", "32000", "64000"]

# ----------------------------- UTILITIES -------------------------------
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
    """Local texture std via Gaussian moments."""
    m1 = gaussian(gray, sigma=sigma, preserve_range=True)
    m2 = gaussian(gray * gray, sigma=sigma, preserve_range=True)
    var = np.clip(m2 - m1 * m1, 0, None)
    std = np.sqrt(var)
    std = std / (std.max() + 1e-8)
    return std

def center_roi_mask(h: int, w: int, margin_frac: float) -> np.ndarray:
    """Centered ROI; trims margins on all sides. margin=0 → full image."""
    r0, r1 = int(h * margin_frac), int(h * (1 - margin_frac))
    c0, c1 = int(w * margin_frac), int(w * (1 - margin_frac))
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m

def mask_scale_bar(h: int, w: int, width_frac: float, height_frac: float, inset_frac: float = 0.02) -> np.ndarray:
    """Mask a bottom-right rectangle (scale bar area)."""
    if width_frac <= 0 or height_frac <= 0:
        return np.zeros((h, w), dtype=bool)
    bw = int(w * width_frac); bh = int(h * height_frac)
    r1 = int(h * (1 - inset_frac)); r0 = max(0, r1 - bh)
    c1 = int(w * (1 - inset_frac)); c0 = max(0, c1 - bw)
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.38, color=(0, 180, 255)) -> np.ndarray:
    out = rgb.copy()
    col = np.zeros_like(out); col[..., 0], col[..., 1], col[..., 2] = color
    out = (out * (1 - alpha) + col * alpha * mask[..., None]).astype(np.uint8)
    return out

def segment_open(gray01: np.ndarray, std_sigma: float, sens: float,
                 open_r: int, close_r: int, min_area: int, open_class: str) -> np.ndarray:
    """
    Texture segmentation. open_class:
      - 'low'  -> open = low texture (recommended for phase contrast)
      - 'high' -> open = high texture
    """
    s = local_std(gray01, sigma=std_sigma)
    thr = threshold_otsu(s)
    thr = thr * (1.0 + sens)  # shift threshold
    mask = (s <= thr) if open_class == "low" else (s >= thr)

    if open_r > 0:
        mask = binary_opening(mask, footprint=disk(open_r))
    if close_r > 0:
        mask = binary_closing(mask, footprint=disk(close_r))
    if min_area > 0:
        mask = remove_small_objects(mask, min_size=min_area)
    return mask

def analyze_image(pil_image: Image.Image, roi_margin: float, bg_sigma: float,
                  std_sigma: float, sens: float, open_r: int, close_r: int,
                  min_area: int, open_class: str, sbw: float, sbh: float):
    """Return (raw_open_pct, overlay_png_bytes)."""
    rgb = np.array(pil_image.convert("RGB"))
    rgb = _downscale(rgb, max_side=1600)
    gray = rgb2gray(rgb).astype(np.float32)

    # illumination correction
    corr = illumination_correct(gray, sigma_bg=bg_sigma)

    # segmentation in texture space
    mask_open = segment_open(corr, std_sigma=std_sigma, sens=sens,
                             open_r=open_r, close_r=close_r, min_area=min_area,
                             open_class=open_class)

    # ROI (center) & optional scale-bar mask (bottom-right)
    h, w = mask_open.shape
    roi = center_roi_mask(h, w, roi_margin)
    sb = mask_scale_bar(h, w, width_frac=sbw, height_frac=sbh)
    keep = roi & ~sb
    keep_pix = int(keep.sum())
    valid = mask_open & keep

    raw_open_pct = 100.0 * (valid.sum() / max(1, keep_pix))

    # Overlay: corrected image + mask + ROI box; gray out scale-bar region
    base = (corr * 255).astype(np.uint8)
    base_rgb = np.repeat(base[..., None], 3, axis=2)
    overlay = overlay_mask(base_rgb, valid, alpha=0.42, color=(0, 180, 255))

    rr0, rr1 = int(h * roi_margin), int(h * (1 - roi_margin))
    cc0, cc1 = int(w * roi_margin), int(w * (1 - roi_margin))
    overlay[rr0:rr1, [cc0, cc1 - 1]] = (0, 255, 90)
    overlay[[rr0, rr1 - 1], cc0:cc1] = (0, 255, 90)
    overlay[sb] = (200, 200, 200)

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

def annotate_bytes(img_bytes: bytes, text: str, corner: str = "br") -> bytes:
    """Draw a semi-transparent label with text onto PNG bytes."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # text block
    pad = 6
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    w, h = img.size
    if corner == "br":   # bottom-right
        x0, y0 = w - tw - 2*pad - 8, h - th - 2*pad - 8
    elif corner == "bl":
        x0, y0 = 8, h - th - 2*pad - 8
    elif corner == "tr":
        x0, y0 = w - tw - 2*pad - 8, 8
    else: # "tl"
        x0, y0 = 8, 8
    x1, y1 = x0 + tw + 2*pad, y0 + th + 2*pad
    draw.rectangle([x0, y0, x1, y1], fill=(0,0,0,150))
    draw.text((x0+pad, y0+pad), text, fill=(255,255,255,255), font=font)
    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()

# ------------------------------- UI -----------------------------------
st.title("Migration Image Analysis")

# --- Sidebar controls (clean main view) ---
with st.sidebar:
    st.header("Settings")
    concentration = st.selectbox("Concentration", CONCENTRATIONS, index=1)
    roi_margin = st.slider("ROI margin (0 = full image)", 0.00, 0.25, 0.00, 0.01,
                           help="Crop edges to avoid artefacts; set 0 for full-frame analysis")
    bg_sigma = st.slider("BG sigma", 10.0, 80.0, 42.0, 2.0, help="Illumination correction blur")
    std_sigma = st.slider("Texture sigma", 3.0, 30.0, 12.0, 1.0, help="Neighborhood size for local std")
    sens = st.slider("Sensitivity", -0.35, 0.35, 0.00, 0.01, help="Lower→more open, Higher→less open")
    cleanup_label = st.selectbox("Cleanup", ["Light (2/2)","Med (3/3)","Strong (4/3)","Custom (1/1)"], index=1)
    map_r = {"Light (2/2)":(2,2), "Med (3/3)":(3,3), "Strong (4/3)":(4,3), "Custom (1/1)":(1,1)}
    open_r, close_r = map_r[cleanup_label]
    open_mode = st.selectbox("Open class", ["Low texture","High texture","Auto"], index=0,
                             help="If results look inverted, try 'High' or 'Auto'")
    sb_mask = st.checkbox("Mask scale bar", value=False)
    sb_width = st.slider("Scale-bar width (frac)", 0.00, 0.30, 0.12, 0.01, disabled=not sb_mask)
    sb_height = st.slider("Scale-bar height (frac)", 0.00, 0.20, 0.06, 0.01, disabled=not sb_mask)

    with st.expander("What do these settings do?"):
        st.markdown("""
- **ROI margin**: trims the image edges before measuring. Set **0** for full-image analysis.
- **BG sigma**: smoothness of illumination correction; larger = stronger background flattening.
- **Texture sigma**: neighborhood size for texture (local std). Larger = coarser texture.
- **Sensitivity**: shifts the threshold on the texture map. **Lower** finds **more** open area; **higher** finds **less**.
- **Cleanup**: morphological open/close radii to remove small noise and smooth boundaries.
- **Open class**: whether open area is **low** texture (typical) or **high** texture (if inverted). **Auto** picks for you.
- **Mask scale bar**: exclude a bottom-right rectangle (set width/height fractions) from measurement.
        """)

# Uploads row (compact grid)
st.markdown("#### Upload images")
u1, u2, u3, u4 = st.columns(4)
uploads = {}
for t, col in zip(TIMEPOINTS, [u1, u2, u3, u4]):
    with col:
        f = st.file_uploader(f"{t} h", type=["png","jpg","jpeg"], key=f"tp{t}", label_visibility="visible")
        if f:
            try:
                img = Image.open(f).convert("RGB")
                uploads[t] = img
                st.image(img, caption=f"{t}h", use_container_width=True)
            except Exception:
                st.error("Could not read image.")

st.divider()
go = st.button("▶️ Analyze", type="primary", use_container_width=True)

# ------------------------------ ANALYSIS ------------------------------
if go and uploads:
    raw = {}
    overlays = {}
    # Initial mode; 'Auto' starts as 'low' then flips if needed
    chosen_mode = "low" if open_mode == "Low texture" else ("high" if open_mode == "High texture" else "low")

    for t in sorted(uploads.keys()):
        val, ov = analyze_image(
            uploads[t],
            roi_margin=roi_margin, bg_sigma=bg_sigma,
            std_sigma=std_sigma, sens=sens,
            open_r=open_r, close_r=close_r, min_area=600,
            open_class=chosen_mode,
            sbw=(sb_width if sb_mask else 0.0), sbh=(sb_height if sb_mask else 0.0)
        )
        raw[t], overlays[t] = val, ov

    # AUTO: flip to 'high' if later timepoints are mostly more open than baseline
    if open_mode == "Auto" and 0 in raw and len(raw) > 1:
        later = [raw[t] for t in raw if t != 0 and not np.isnan(raw[t])]
        if later and np.nanmedian(later) > raw[0]:
            raw.clear(); overlays.clear()
            for t in sorted(uploads.keys()):
                val, ov = analyze_image(
                    uploads[t],
                    roi_margin=roi_margin, bg_sigma=bg_sigma,
                    std_sigma=std_sigma, sens=sens,
                    open_r=open_r, close_r=close_r, min_area=600,
                    open_class="high",
                    sbw=(sb_width if sb_mask else 0.0), sbh=(sb_height if sb_mask else 0.0)
                )
                raw[t], overlays[t] = val, ov

    # Build baseline-normalized table
    df, baseline = summarize_series(raw)

    # Annotate overlays with metrics and display in a compact grid
    st.markdown("#### Detection overlays (annotated)")
    gcols = st.columns(2)
    for i, t in enumerate(sorted(overlays.keys())):
        rel = df.loc[t, "Relative Open %"] if t in df.index else np.nan
        clo = df.loc[t, "Closure %"] if t in df.index else np.nan
        if np.isnan(rel):
            label = f"{t}h — Open {raw[t]:.2f}%"
        else:
            label = f"{t}h — Open {raw[t]:.2f}% | Rel {rel:.1f}% | Close {clo:.1f}%"
        annotated = annotate_bytes(overlays[t], label, corner="br")
        with gcols[i % 2]:
            st.image(annotated, caption=None, use_container_width=True)

    # Table + plot + CSV in a right column
    st.markdown("#### 📊 Baseline-normalized results")
    st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

    # Closure plot with dynamic y-limits
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = df.index.values
    y = df["Closure %"].values.astype(float)
    if np.isfinite(y).any():
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        ylo = min(-10, ymin - 5) if np.isfinite(ymin) else -10
        yhi = max(100, ymax + 5) if np.isfinite(ymax) else 100
    else:
        ylo, yhi = -10, 100
    ax.plot(x, y, marker="o", linewidth=2, color="#009E73", label=f"{concentration} p/mL")
    ax.set_xlabel("Hours"); ax.set_ylabel("Closure % (relative to 0h)")
    ax.set_title(f"Closure Curve — {concentration} p/mL (baseline {baseline:.2f}% open)")
    ax.grid(True, linestyle="--", alpha=0.5); ax.set_ylim(ylo, yhi)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # CSV download
    csv = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name=f"results_{concentration}.csv", use_container_width=True)

else:
    st.info("Upload at least one image and click **Analyze**. For baseline normalization, include **0h**.")

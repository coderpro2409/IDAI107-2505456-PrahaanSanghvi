"""CurbSense Lite - upload a parking-lot photo, see free vs occupied bays and a quick recommendation."""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

WEIGHTS = Path("bay_detector.pt")
FREE_COLOUR, BUSY_COLOUR = (56, 189, 248), (249, 115, 22)    # blue = free, orange = occupied


def crowding(pct: float) -> str:
    return "Light" if pct < 40 else "Heavy" if pct > 75 else "Steady"


def advice(free: int, total: int, rating: str) -> str:
    if total == 0:
        return "No bays found. Try a clearer photo or a lower threshold."
    if free == 0:
        return "Lot is full. Look elsewhere."
    noun = "bay" if free == 1 else "bays"
    return {"Heavy": f"Only {free} {noun} free. Another lot may be better.",
            "Steady": f"{free} {noun} free. Go ahead, but it is filling up.",
            "Light": f"{free} {noun} free. Plenty of room."}[rating]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


st.set_page_config(page_title="CurbSense Lite", page_icon="🅿️")
st.title("🅿️ CurbSense Lite")
st.caption("🔵 free bay · 🟠 occupied bay")

if not WEIGHTS.exists():
    st.error("Put your trained weights next to this file as `bay_detector.pt`.")
    st.stop()

threshold = st.slider("Detection threshold", 0.10, 0.90, 0.40, 0.05)
file = st.file_uploader("Parking-lot photo", type=["jpg", "jpeg", "png"])
if not file:
    st.stop()

photo = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
model = load_model()
boxes = model.predict(photo, conf=threshold, imgsz=640, verbose=False)[0].boxes

base = np.asarray(photo).copy()
tint, free = base.copy(), 0
found = []
for b in boxes:
    is_free = "free" in str(model.names[int(b.cls)]).lower()
    free += is_free
    found.append((is_free, *map(int, b.xyxy[0].tolist())))
for is_free, x1, y1, x2, y2 in found:
    cv2.rectangle(tint, (x1, y1), (x2, y2), FREE_COLOUR if is_free else BUSY_COLOUR, cv2.FILLED)
out = cv2.addWeighted(tint, 0.3, base, 0.7, 0)
for is_free, x1, y1, x2, y2 in found:
    cv2.rectangle(out, (x1, y1), (x2, y2), FREE_COLOUR if is_free else BUSY_COLOUR, 2)

total = len(found)
pct = round(100 * (total - free) / total, 1) if total else 0.0
rating = crowding(pct)

st.subheader(f"Crowding: {rating}")
st.info(advice(free, total, rating))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bays", total)
c2.metric("Free", free)
c3.metric("Occupied", total - free)
c4.metric("Occupancy", f"{pct}%")
st.image(out, width="stretch")
ok, png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
if ok:
    st.download_button("Download result", png.tobytes(), "result.png", "image/png")

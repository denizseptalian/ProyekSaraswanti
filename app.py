"""
Deteksi Objek YOLOv8 TFLite - Streamlit Cloud
Mode: Live Webcam (WebRTC), Upload Video, Upload Gambar
"""
import av
import cv2
import numpy as np
import streamlit as st
import tempfile
import time
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ================= Konfigurasi =================
MODEL_PATH = Path(__file__).parent / "best_model.tflite"
INPUT_SIZE = 640

# Ganti sesuai kelas pada dataset Anda (urutan harus sama dengan data.yaml Roboflow)
CLASS_NAMES = ["kelas_0", "kelas_1", "kelas_2", "kelas_3", "kelas_4", "kelas_5"]

COLORS = [
    (56, 176, 0), (255, 89, 94), (25, 130, 196),
    (255, 202, 58), (106, 76, 147), (255, 121, 63),
]

st.set_page_config(page_title="Deteksi YOLOv8 TFLite", page_icon="🌴", layout="wide")


# ================= Load Model =================
@st.cache_resource
def load_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(MODEL_PATH), num_threads=4)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


interpreter, input_details, output_details = load_interpreter()
n_classes = output_details["shape"][1] - 4
if len(CLASS_NAMES) != n_classes:
    CLASS_NAMES = [f"kelas_{i}" for i in range(n_classes)]


# ================= Pre/Post-processing =================
def letterbox(img, size=INPUT_SIZE):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = (size - nh) / 2, (size - nw) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def detect(frame_bgr, conf_thres=0.35, iou_thres=0.45):
    """Jalankan inferensi YOLOv8 TFLite pada satu frame BGR."""
    img, r, pad_x, pad_y = letterbox(frame_bgr)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None]  # NCHW

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details["index"])[0]  # (4+nc, 8400)
    pred = pred.T  # (8400, 4+nc)

    boxes_xywh = pred[:, :4] * INPUT_SIZE  # koordinat ternormalisasi -> piksel input
    scores_all = pred[:, 4:]
    class_ids = np.argmax(scores_all, axis=1)
    confidences = scores_all[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thres
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    # xywh (center) -> xyxy pada koordinat frame asli
    cx, cy, bw, bh = boxes_xywh.T
    x1 = (cx - bw / 2 - pad_x) / r
    y1 = (cy - bh / 2 - pad_y) / r
    x2 = (cx + bw / 2 - pad_x) / r
    y2 = (cy + bh / 2 - pad_y) / r

    h, w = frame_bgr.shape[:2]
    x1, y1 = np.clip(x1, 0, w), np.clip(y1, 0, h)
    x2, y2 = np.clip(x2, 0, w), np.clip(y2, 0, h)

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_thres, iou_thres)
    if len(keep) == 0:
        return []
    keep = np.array(keep).flatten()

    return [
        dict(box=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
             conf=float(confidences[i]), cls=int(class_ids[i]))
        for i in keep
    ]


def draw(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["cls"] % len(COLORS)]
        label = f'{CLASS_NAMES[d["cls"]]} {d["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ================= UI =================
st.title("🌴 Deteksi Objek YOLOv8 (TFLite)")
st.caption("Model: best_model.tflite · Input 640×640 · Berjalan di CPU Streamlit Cloud")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    skip_n = st.slider("Proses setiap N frame (webcam)", 1, 5, 2,
                       help="Nilai lebih besar = lebih lancar di CPU, deteksi diperbarui lebih jarang")
    st.divider()
    st.markdown("**Kelas model:**")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(f"- `{i}`: {name}")

tab_cam, tab_video, tab_img = st.tabs(["📹 Live Webcam", "🎞️ Upload Video", "🖼️ Upload Gambar"])

# ---------- Tab 1: Live Webcam (WebRTC) ----------
with tab_cam:
    st.markdown("Izinkan akses kamera di browser. Deteksi berjalan real-time pada stream.")

    class Processor:
        def __init__(self):
            self.count = 0
            self.last_dets = []

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.count += 1
            if self.count % skip_n == 0 or not self.last_dets:
                self.last_dets = detect(img, conf_thres, iou_thres)
            img = draw(img, self.last_dets)
            cv2.putText(img, f"Objek: {len(self.last_dets)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="yolo-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=Processor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

# ---------- Tab 2: Upload Video ----------
with tab_video:
    vid_file = st.file_uploader("Unggah video (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())

        col1, col2 = st.columns([3, 1])
        frame_slot = col1.empty()
        stats_slot = col2.empty()
        progress = st.progress(0, text="Memproses video...")

        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

        # tulis video hasil untuk diunduh
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

        i, t0, all_counts = 0, time.time(), []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(frame, conf_thres, iou_thres)
            all_counts.append(len(dets))
            frame = draw(frame, dets)
            writer.write(frame)

            i += 1
            if i % 3 == 0:  # update tampilan tiap 3 frame agar UI tidak berat
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)
                elapsed = time.time() - t0
                stats_slot.metric("FPS proses", f"{i/elapsed:.1f}")
                stats_slot.metric("Deteksi frame ini", len(dets))
            progress.progress(min(i / total, 1.0), text=f"Frame {i}/{total}")

        cap.release()
        writer.release()
        progress.progress(1.0, text="Selesai ✅")

        st.success(f"Selesai: {i} frame · rata-rata {np.mean(all_counts):.1f} objek/frame · "
                   f"maks {max(all_counts) if all_counts else 0} objek")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Unduh video hasil deteksi", f,
                               file_name="hasil_deteksi.mp4", mime="video/mp4")

# ---------- Tab 3: Upload Gambar ----------
with tab_img:
    img_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        data = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        t0 = time.time()
        dets = detect(frame, conf_thres, iou_thres)
        ms = (time.time() - t0) * 1000
        frame = draw(frame, dets)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.info(f"⏱️ Inferensi: {ms:.0f} ms · 🎯 {len(dets)} objek terdeteksi")
        if dets:
            st.dataframe(
                [{"Kelas": CLASS_NAMES[d["cls"]], "Confidence": f'{d["conf"]:.2%}',
                  "Box (x1,y1,x2,y2)": str(d["box"])} for d in dets],
                use_container_width=True,
            )

"""
Deteksi & Tracking YOLOv8 TFLite – Ripeness Palm Oil
Kelas: Abnormal | Jangkos | masak | mengkal | mentah | overipe
"""
import av
import cv2
import numpy as np
import streamlit as st
import tempfile
import time
import threading
from pathlib import Path
from collections import defaultdict
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ═══════════════════════════════════════════════════════
# Konfigurasi
# ═══════════════════════════════════════════════════════
MODEL_PATH = Path(__file__).parent / "best_model.tflite"
INPUT_SIZE  = 640

CLASS_NAMES = ["Abnormal", "Jangkos", "masak", "mengkal", "mentah", "overipe"]

# Warna per kelas (BGR)
CLASS_COLORS = [
    (0,   50, 220),   # Abnormal  – merah
    (0,  140, 255),   # Jangkos   – oranye
    (30, 200,  80),   # masak     – hijau
    (0,  220, 200),   # mengkal   – kuning-hijau
    (200, 200,  0),   # mentah    – cyan
    (160,  0, 200),   # overipe   – ungu
]

st.set_page_config(page_title="Ripeness Detector – PT SAE", page_icon="🌴", layout="wide")

# ═══════════════════════════════════════════════════════
# Load Model (cache agar hanya load sekali)
# ═══════════════════════════════════════════════════════
@st.cache_resource
def load_interpreter():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(MODEL_PATH), num_threads=4)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

interpreter, input_details, output_details = load_interpreter()
n_classes = output_details["shape"][1] - 4  # seharusnya 6

# ═══════════════════════════════════════════════════════
# Pre/Post-processing
# ═══════════════════════════════════════════════════════
def letterbox(img, size=INPUT_SIZE):
    h, w = img.shape[:2]
    r    = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = (size - nh) / 2, (size - nw) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top  = int(round(pad_h - 0.1)); bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1)); right  = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def detect(frame_bgr, conf_thres=0.35, iou_thres=0.45):
    img, r, pad_x, pad_y = letterbox(frame_bgr)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None]

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details["index"])[0]  # (4+nc, 8400)
    pred = pred.T                                               # (8400, 4+nc)

    boxes_xywh  = pred[:, :4] * INPUT_SIZE
    scores_all  = pred[:, 4:]
    class_ids   = np.argmax(scores_all, axis=1)
    confidences = scores_all[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thres
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    cx, cy, bw, bh = boxes_xywh.T
    h, w = frame_bgr.shape[:2]
    x1 = np.clip((cx - bw / 2 - pad_x) / r, 0, w)
    y1 = np.clip((cy - bh / 2 - pad_y) / r, 0, h)
    x2 = np.clip((cx + bw / 2 - pad_x) / r, 0, w)
    y2 = np.clip((cy + bh / 2 - pad_y) / r, 0, h)

    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_thres, iou_thres)
    if len(keep) == 0:
        return []
    keep = np.array(keep).flatten()

    return [dict(box=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                 conf=float(confidences[i]),
                 cls=int(class_ids[i]))
            for i in keep]

# ═══════════════════════════════════════════════════════
# Tracker (IoU matching – 1 objek dihitung 1 kali)
# ═══════════════════════════════════════════════════════
def _iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """
    Setiap objek unik mendapat ID.
    Setelah terdeteksi konsisten min_hits frame → dihitung 1× (counted=True).
    Jika objek hilang > max_age frame → track dihapus.
    """
    def __init__(self, max_age=20, min_hits=4, iou_thresh=0.25):
        self.tracks: dict = {}   # tid → {box, cls, conf, age, hits, counted}
        self.next_id   = 0
        self.max_age   = max_age
        self.min_hits  = min_hits
        self.iou_thresh = iou_thresh
        self.class_counts: dict = defaultdict(int)
        self.total_counted = 0
        self._lock = threading.Lock()

    def update(self, detections):
        with self._lock:
            for t in self.tracks.values():
                t['_seen'] = False

            for det in detections:
                best_iou, best_id = self.iou_thresh, None
                for tid, t in self.tracks.items():
                    if t['cls'] != det['cls']:
                        continue
                    score = _iou(det['box'], t['box'])
                    if score > best_iou:
                        best_iou, best_id = score, tid

                if best_id is not None:
                    t = self.tracks[best_id]
                    t['box']   = det['box']
                    t['conf']  = det['conf']
                    t['age']   = 0
                    t['hits'] += 1
                    t['_seen'] = True
                    if t['hits'] >= self.min_hits and not t['counted']:
                        t['counted'] = True
                        self.class_counts[det['cls']] += 1
                        self.total_counted += 1
                else:
                    self.tracks[self.next_id] = dict(
                        box=det['box'], cls=det['cls'], conf=det['conf'],
                        age=0, hits=1, counted=False, _seen=True)
                    self.next_id += 1

            for tid, t in self.tracks.items():
                if not t['_seen']:
                    t['age'] += 1
            dead = [tid for tid, t in self.tracks.items()
                    if not t['_seen'] and t['age'] > self.max_age]
            for tid in dead:
                del self.tracks[tid]

            return dict(self.tracks)

    def snapshot_counts(self):
        with self._lock:
            return dict(self.class_counts), self.total_counted

    def reset(self):
        with self._lock:
            self.tracks.clear()
            self.class_counts = defaultdict(int)
            self.total_counted = 0
            self.next_id = 0

# ═══════════════════════════════════════════════════════
# Draw
# ═══════════════════════════════════════════════════════
def _dashed_rect(img, x1, y1, x2, y2, color, dash=12, gap=6):
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(img, (x, y1), (min(x+dash, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x+dash, x2), y2), color, 2)
    for y in range(y1, y2, step):
        cv2.line(img, (x1, y), (x1, min(y+dash, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y+dash, y2)), color, 2)


def draw_tracks(frame, tracks):
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t['box']
        color   = CLASS_COLORS[t['cls'] % len(CLASS_COLORS)]
        label   = f"#{tid} {CLASS_NAMES[t['cls']]} {t['conf']:.2f}"
        counted = t['counted']

        if counted:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label += "  ✓"
        else:
            _dashed_rect(frame, x1, y1, x2, y2, color)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_simple(frame, detections):
    """Untuk gambar statis – tanpa tracking."""
    for d in detections:
        x1, y1, x2, y2 = d['box']
        color = CLASS_COLORS[d['cls'] % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[d['cls']]}  {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_hud(frame, counts: dict, total: int):
    """Overlay counter kumulatif di pojok kiri atas."""
    rows = [f"TOTAL: {total}"] + [f"{CLASS_NAMES[c]}: {n}"
                                    for c, n in sorted(counts.items())]
    pad, line_h = 8, 22
    box_h = pad * 2 + line_h * len(rows)
    box_w = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, row in enumerate(rows):
        color = (0, 255, 255) if i == 0 else CLASS_COLORS[sorted(counts.keys())[i-1] % len(CLASS_COLORS)] if i > 0 and i - 1 < len(counts) else (200, 200, 200)
        cv2.putText(frame, row, (pad, pad + line_h * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

# ═══════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════
st.title("🌴 Ripeness Detector – Palm Oil (YOLOv8 TFLite)")
st.caption("Tracking per-objek · dihitung 1× · PT SAE / PT SSM")

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres  = st.slider("IoU NMS threshold",    0.10, 0.90, 0.45, 0.05)
    min_hits   = st.slider("Min hits sebelum dihitung (frame)", 1, 8, 4,
                           help="Objek dihitung setelah terdeteksi N frame berturut-turut")
    skip_n     = st.slider("Proses setiap N frame (webcam)", 1, 5, 2)
    st.divider()
    st.markdown("**Legend:**")
    for i, name in enumerate(CLASS_NAMES):
        rgb = CLASS_COLORS[i][::-1]
        st.markdown(f'<span style="color:rgb{rgb}">■</span> `{name}`',
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("**Arti box:**")
    st.markdown("` ╌╌ ` putus-putus = sedang dilacak")
    st.markdown("`──` solid + ✓  = sudah dihitung")

tab_cam, tab_video, tab_img = st.tabs(
    ["📹 Live Webcam", "🎞️ Upload Video", "🖼️ Upload Gambar"])

# ═══════════════════════════════════════════════════════
# TAB 1 – Webcam Live
# ═══════════════════════════════════════════════════════
with tab_cam:
    st.markdown("Kamera aktif → objek dilacak & dihitung otomatis. "
                "Box putus-putus = baru masuk frame. Box solid ✓ = sudah terhitung.")

    # Shared tracker (persistent selama session)
    if "tracker" not in st.session_state:
        st.session_state.tracker = ObjectTracker(max_age=20,
                                                  min_hits=min_hits,
                                                  iou_thresh=0.25)
    tracker: ObjectTracker = st.session_state.tracker
    tracker.min_hits = min_hits  # ikuti slider

    col_cam, col_cnt = st.columns([3, 1])
    count_slot = col_cnt.empty()

    class LiveProcessor:
        def __init__(self):
            self.frame_n = 0
            self.last_dets = []

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_n += 1
            if self.frame_n % skip_n == 0:
                self.last_dets = detect(img, conf_thres, iou_thres)
            active = tracker.update(self.last_dets)
            counts, total = tracker.snapshot_counts()
            img = draw_tracks(img, active)
            img = draw_hud(img, counts, total)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with col_cam:
        ctx = webrtc_streamer(
            key="yolo-track",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480},
                                      "audio": False},
            async_processing=True,
        )

    # Tampilan counter real-time (polling)
    if ctx.state.playing:
        while ctx.state.playing:
            counts, total = tracker.snapshot_counts()
            with count_slot.container():
                st.metric("Total objek unik", total)
                for cls_id, n in sorted(counts.items()):
                    st.metric(CLASS_NAMES[cls_id], n)
            time.sleep(0.5)

    if st.button("🔄 Reset Counter", key="rst_cam"):
        tracker.reset()
        st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 2 – Upload Video
# ═══════════════════════════════════════════════════════
with tab_video:
    vid_file = st.file_uploader("Unggah video (mp4/avi/mov)",
                                type=["mp4", "avi", "mov", "mkv"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())

        vid_tracker = ObjectTracker(max_age=20, min_hits=min_hits, iou_thresh=0.25)

        col_v, col_s = st.columns([3, 1])
        frame_slot  = col_v.empty()
        stat_slot   = col_s.empty()
        progress    = st.progress(0, text="Memproses…")

        cap   = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer   = cv2.VideoWriter(out_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0; t0 = time.time()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets   = detect(frame, conf_thres, iou_thres)
            active = vid_tracker.update(dets)
            counts, tot = vid_tracker.snapshot_counts()
            frame  = draw_tracks(frame, active)
            frame  = draw_hud(frame, counts, tot)
            writer.write(frame)
            i += 1
            if i % 4 == 0:
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)
                with stat_slot.container():
                    st.metric("Frame", f"{i}/{total}")
                    st.metric("FPS proses", f"{i/(time.time()-t0):.1f}")
                    st.metric("Total dihitung", tot)
                    for c, n in sorted(counts.items()):
                        st.metric(CLASS_NAMES[c], n)
            progress.progress(min(i/total, 1.0), text=f"Frame {i}/{total}")

        cap.release(); writer.release()
        progress.progress(1.0, text="Selesai ✅")
        counts, tot = vid_tracker.snapshot_counts()
        st.success(f"Selesai · **{tot} objek unik** terdeteksi dari {i} frame")
        cols = st.columns(len(CLASS_NAMES))
        for idx, (c, n) in enumerate(sorted(counts.items())):
            cols[idx].metric(CLASS_NAMES[c], n)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Unduh video hasil", f,
                               file_name="hasil_deteksi.mp4", mime="video/mp4")

# ═══════════════════════════════════════════════════════
# TAB 3 – Gambar Statis
# ═══════════════════════════════════════════════════════
with tab_img:
    img_file = st.file_uploader("Unggah gambar",
                                type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        data  = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        t0    = time.time()
        dets  = detect(frame, conf_thres, iou_thres)
        ms    = (time.time() - t0) * 1000
        frame = draw_simple(frame, dets)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.info(f"⏱️ {ms:.0f} ms · 🎯 {len(dets)} objek terdeteksi")
        if dets:
            summary = defaultdict(lambda: dict(n=0, conf_sum=0))
            for d in dets:
                summary[d['cls']]['n'] += 1
                summary[d['cls']]['conf_sum'] += d['conf']
            st.dataframe(
                [{"Kelas": CLASS_NAMES[c],
                  "Jumlah": v['n'],
                  "Avg Confidence": f"{v['conf_sum']/v['n']:.2%}"}
                 for c, v in sorted(summary.items())],
                use_container_width=True)

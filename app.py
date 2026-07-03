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

# ═══════════════════════════════════════════════════════════════
# Konfigurasi
# ═══════════════════════════════════════════════════════════════
MODEL_PATH  = Path(__file__).parent / "best_model.tflite"
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

# Warna hex untuk UI Streamlit (RGB)
CLASS_HEX = ["#dc3232", "#ff8c00", "#1ec850", "#c8dc00", "#00c8c8", "#a000c8"]

st.set_page_config(
    page_title="Ripeness Detector – PT SAE",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS: fullscreen-friendly, mobile-first
st.markdown("""
<style>
/* sembunyikan padding atas bawaan streamlit */
.block-container { padding-top: 1rem !important; }
/* kartu counter */
.cls-card {
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
}
.cls-card .cls-count {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}
.total-card {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 14px;
    background: #1a1a2e;
    border: 2px solid #4fc3f7;
    text-align: center;
}
.total-card .lbl  { color: #90caf9; font-size: 0.85rem; letter-spacing: 1px; }
.total-card .num  { color: #4fc3f7; font-size: 2.4rem; font-weight: 900; line-height:1; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Load Model
# ═══════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════
# Pre / Post-processing
# ═══════════════════════════════════════════════════════════════
def letterbox(img, size=INPUT_SIZE):
    h, w  = img.shape[:2]
    r     = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    ph, pw = (size - nh) / 2, (size - nw) / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(ph - 0.1)), int(round(ph + 0.1))
    left, right = int(round(pw - 0.1)), int(round(pw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def detect(frame_bgr, conf_thres=0.35, iou_thres=0.45):
    img, r, px, py = letterbox(frame_bgr)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None]

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details["index"])[0].T  # (8400, 4+nc)

    boxes_xywh  = pred[:, :4] * INPUT_SIZE
    scores_all  = pred[:, 4:]
    class_ids   = np.argmax(scores_all, axis=1)
    confs       = scores_all[np.arange(len(class_ids)), class_ids]

    mask = confs >= conf_thres
    boxes_xywh, confs, class_ids = boxes_xywh[mask], confs[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    cx, cy, bw, bh = boxes_xywh.T
    h, w = frame_bgr.shape[:2]
    x1 = np.clip((cx - bw/2 - px) / r, 0, w)
    y1 = np.clip((cy - bh/2 - py) / r, 0, h)
    x2 = np.clip((cx + bw/2 - px) / r, 0, w)
    y2 = np.clip((cy + bh/2 - py) / r, 0, h)

    nms_in = np.stack([x1, y1, x2-x1, y2-y1], axis=1).tolist()
    keep   = cv2.dnn.NMSBoxes(nms_in, confs.tolist(), conf_thres, iou_thres)
    if len(keep) == 0:
        return []
    keep = np.array(keep).flatten()
    return [dict(box=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                 conf=float(confs[i]), cls=int(class_ids[i])) for i in keep]

# ═══════════════════════════════════════════════════════════════
# Tracker
# ═══════════════════════════════════════════════════════════════
def _iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    def __init__(self, max_age=20, min_hits=4, iou_thresh=0.25):
        self.tracks        = {}
        self.next_id       = 0
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_thresh    = iou_thresh
        self.class_counts  = defaultdict(int)
        self.total_counted = 0
        self._lock         = threading.Lock()

    def update(self, detections):
        with self._lock:
            for t in self.tracks.values():
                t['_seen'] = False

            for det in detections:
                best_iou, best_id = self.iou_thresh, None
                for tid, t in self.tracks.items():
                    if t['cls'] != det['cls']:
                        continue
                    s = _iou(det['box'], t['box'])
                    if s > best_iou:
                        best_iou, best_id = s, tid

                if best_id is not None:
                    t = self.tracks[best_id]
                    t.update(box=det['box'], conf=det['conf'],
                             age=0, _seen=True)
                    t['hits'] += 1
                    if t['hits'] >= self.min_hits and not t['counted']:
                        t['counted'] = True
                        self.class_counts[det['cls']] += 1
                        self.total_counted += 1
                else:
                    self.tracks[self.next_id] = dict(
                        box=det['box'], cls=det['cls'], conf=det['conf'],
                        age=0, hits=1, counted=False, _seen=True)
                    self.next_id += 1

            for t in self.tracks.values():
                if not t['_seen']:
                    t['age'] += 1
            dead = [tid for tid, t in self.tracks.items()
                    if not t['_seen'] and t['age'] > self.max_age]
            for tid in dead:
                del self.tracks[tid]

            return dict(self.tracks)

    def snapshot(self):
        with self._lock:
            return dict(self.class_counts), self.total_counted

    def reset(self):
        with self._lock:
            self.tracks.clear()
            self.class_counts = defaultdict(int)
            self.total_counted = 0
            self.next_id = 0

# ================================================================
# Draw
# ================================================================
def _dashed_rect(img, x1, y1, x2, y2, color, dash=14, gap=7):
    step = dash + gap
    for x in range(x1, x2, step):
        cv2.line(img, (x, y1),  (min(x+dash, x2), y1),  color, 2)
        cv2.line(img, (x, y2),  (min(x+dash, x2), y2),  color, 2)
    for y in range(y1, y2, step):
        cv2.line(img, (x1, y),  (x1, min(y+dash, y2)),  color, 2)
        cv2.line(img, (x2, y),  (x2, min(y+dash, y2)),  color, 2)


def draw_tracks(frame, tracks):
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t["box"]
        color   = CLASS_COLORS[t["cls"] % len(CLASS_COLORS)]
        name    = CLASS_NAMES[t["cls"]]
        counted = t["counted"]
        label   = f"#{tid} {name} {t['conf']:.2f}" + (" v" if counted else "")

        if counted:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        else:
            _dashed_rect(frame, x1, y1, x2, y2, color)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_simple(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = CLASS_COLORS[d["cls"] % len(CLASS_COLORS)]
        label = f"{CLASS_NAMES[d['cls']]}  {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_hud(frame, counts, total):
    """Panel semitransparent sudut kanan bawah: semua kelas selalu tampil.
    Count bertambah hanya saat objek baru dikonfirmasi tracker."""
    h, w   = frame.shape[:2]
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 10
    lh     = 28     # line height
    dr     = 7      # dot radius
    pw     = 230    # panel width
    n_rows = len(CLASS_NAMES) + 2
    ph     = pad * 2 + lh * n_rows + 6

    x0, y0 = w - pw - 10, h - ph - 10

    # Background semitransparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+pw, y0+ph), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0+pw, y0+ph), (70, 70, 70), 1)

    # Judul
    y = y0 + pad + 18
    cv2.putText(frame, "RIPENESS COUNTER", (x0+pad, y),
                FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Setiap kelas
    for i, name in enumerate(CLASS_NAMES):
        y    += lh
        cnt   = counts.get(i, 0)
        color = CLASS_COLORS[i]
        active = cnt > 0

        cv2.circle(frame, (x0+pad+dr, y-5), dr,
                   color if active else (55, 55, 55), -1)

        txt_col = (230, 230, 230) if active else (110, 110, 110)
        cv2.putText(frame, name, (x0+pad+dr*2+6, y),
                    FONT, 0.50, txt_col, 1, cv2.LINE_AA)

        cnt_str = str(cnt)
        (cw, _), _ = cv2.getTextSize(cnt_str, FONT, 0.62, 2)
        cnt_col = color if active else (70, 70, 70)
        cv2.putText(frame, cnt_str, (x0+pw-pad-cw, y),
                    FONT, 0.62, cnt_col, 2, cv2.LINE_AA)

    # Divider + Total
    y += 10
    cv2.line(frame, (x0+pad, y), (x0+pw-pad, y), (80, 80, 80), 1)
    y += lh - 2
    cv2.putText(frame, "TOTAL", (x0+pad+dr*2+6, y),
                FONT, 0.52, (0, 220, 220), 1, cv2.LINE_AA)
    ts = str(total)
    (tw, _), _ = cv2.getTextSize(ts, FONT, 0.72, 2)
    cv2.putText(frame, ts, (x0+pw-pad-tw, y),
                FONT, 0.72, (0, 220, 220), 2, cv2.LINE_AA)

    return frame

# ═══════════════════════════════════════════════════════════════
# Helper UI: render counter cards di PAGE (bukan di frame)
# ═══════════════════════════════════════════════════════════════
def render_counts(container, counts: dict, total: int):
    """Tampilkan kartu hitungan berwarna di Streamlit container."""
    with container:
        st.markdown(f"""
        <div class="total-card">
          <div class="lbl">TOTAL OBJEK UNIK</div>
          <div class="num">{total}</div>
        </div>""", unsafe_allow_html=True)

        for cls_id in range(len(CLASS_NAMES)):
            n   = counts.get(cls_id, 0)
            hex_c = CLASS_HEX[cls_id]
            st.markdown(f"""
            <div class="cls-card" style="background:{hex_c}cc; border-left: 5px solid {hex_c};">
              <span>{CLASS_NAMES[cls_id]}</span>
              <span class="cls-count">{n}</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════
st.title("🌴 Palm Oil Ripeness Detector")
st.caption("YOLOv8 TFLite · Tracking per-objek · dihitung 1× · PT SAE / PT SSM")

# Sidebar pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan")
    conf_thres = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thres  = st.slider("IoU NMS threshold",    0.10, 0.90, 0.45, 0.05)
    min_hits   = st.slider("Min hits sebelum dihitung", 1, 8, 4,
                           help="Objek dihitung setelah muncul N frame berturut-turut")
    skip_n     = st.slider("Proses setiap N frame", 1, 4, 2,
                           help="Lebih tinggi = lebih lancar, update lebih jarang")
    st.divider()
    st.markdown("**Legend warna:**")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(
            f'<span style="background:{CLASS_HEX[i]};border-radius:4px;'
            f'padding:2px 8px;color:#fff;font-weight:600">{name}</span>',
            unsafe_allow_html=True)
    st.markdown("")
    st.markdown("**Box:**  \n`╌╌` putus = dilacak  \n`──✓` solid = dihitung")

tab_cam, tab_video, tab_img = st.tabs(
    ["📹 Kamera Live", "🎞️ Upload Video", "🖼️ Upload Gambar"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 – Webcam Live (mobile-optimized)
# ═══════════════════════════════════════════════════════════════
with tab_cam:
    # Inisialisasi tracker di session_state supaya persistent
    if "cam_tracker" not in st.session_state:
        st.session_state.cam_tracker = ObjectTracker(
            max_age=25, min_hits=min_hits, iou_thresh=0.25)
    tracker: ObjectTracker = st.session_state.cam_tracker
    tracker.min_hits = min_hits

    # Layout: video di kiri (lebar), counter di kanan
    col_vid, col_cnt = st.columns([3, 1], gap="medium")

    with col_cnt:
        st.markdown("#### 📊 Hitungan")
        count_placeholder = st.empty()
        st.markdown("---")
        if st.button("🔄 Reset Counter", use_container_width=True, key="rst_cam"):
            tracker.reset()
            st.rerun()
        st.markdown("""
        <small>
        ╌╌ = <b>dilacak</b><br>
        ──✓ = <b>dihitung</b>
        </small>""", unsafe_allow_html=True)

    # Tampilkan counts awal (sebelum kamera aktif)
    counts0, total0 = tracker.snapshot()
    render_counts(count_placeholder, counts0, total0)

    class LiveProcessor:
        def __init__(self):
            self.frame_n   = 0
            self.last_dets = []

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Ambil frame asli tanpa resize – biarkan browser kirim resolusi penuh
            img = frame.to_ndarray(format="bgr24")
            self.frame_n += 1
            if self.frame_n % skip_n == 0 or not self.last_dets:
                self.last_dets = detect(img, conf_thres, iou_thres)
            active = tracker.update(self.last_dets)
            img    = draw_tracks(img, active)
            img    = draw_hud(img, counts, total)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with col_vid:
        ctx = webrtc_streamer(
            key="yolo-cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            }),
            video_processor_factory=LiveProcessor,
            media_stream_constraints={
                "video": {
                    # HP: kamera belakang, resolusi tinggi, framerate wajar
                    "facingMode": {"ideal": "environment"},
                    "width":     {"ideal": 1280, "min": 640},
                    "height":    {"ideal": 720,  "min": 480},
                    "frameRate": {"ideal": 30,   "min": 15},
                },
                "audio": False,
            },
            # Biarkan WebRTC pilih codec terbaik, jangan paksa resolusi output
            async_processing=True,
        )

    # Polling counter: update kartu di page selama kamera aktif
    if ctx.state.playing:
        while ctx.state.playing:
            c, tot = tracker.snapshot()
            render_counts(count_placeholder, c, tot)
            time.sleep(0.4)

# ═══════════════════════════════════════════════════════════════
# TAB 2 – Upload Video
# ═══════════════════════════════════════════════════════════════
with tab_video:
    vid_file = st.file_uploader(
        "Unggah video (mp4 / avi / mov)", type=["mp4", "avi", "mov", "mkv"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())

        vid_tracker = ObjectTracker(max_age=20, min_hits=min_hits, iou_thresh=0.25)

        col_v, col_s = st.columns([3, 1], gap="medium")
        frame_slot   = col_v.empty()

        with col_s:
            st.markdown("#### 📊 Hitungan")
            vid_count_slot = st.empty()
            prog_slot      = st.empty()

        progress = st.progress(0, text="Memproses…")

        cap   = cv2.VideoCapture(tfile.name)
        n_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer   = cv2.VideoWriter(out_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        i  = 0
        t0 = time.time()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets   = detect(frame, conf_thres, iou_thres)
            active = vid_tracker.update(dets)
            c_now, tot_now = vid_tracker.snapshot()
            frame  = draw_tracks(frame, active)
            frame  = draw_hud(frame, c_now, tot_now)
            writer.write(frame)
            i += 1
            if i % 4 == 0:
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)
                c, tot = vid_tracker.snapshot()
                render_counts(vid_count_slot, c, tot)
                prog_slot.caption(f"Frame {i}/{n_frm} · {i/(time.time()-t0):.1f} fps")
            progress.progress(min(i / n_frm, 1.0), text=f"Frame {i}/{n_frm}")

        cap.release()
        writer.release()
        progress.progress(1.0, text="Selesai ✅")

        c, tot = vid_tracker.snapshot()
        st.success(f"Selesai · **{tot} objek unik** terdeteksi dari {i} frame")

        # Ringkasan akhir
        cols = st.columns(len(CLASS_NAMES))
        for idx, name in enumerate(CLASS_NAMES):
            cols[idx].metric(name, c.get(idx, 0))

        with open(out_path, "rb") as f:
            st.download_button("⬇️ Unduh video hasil", f,
                               file_name="hasil_deteksi.mp4", mime="video/mp4")

# ═══════════════════════════════════════════════════════════════
# TAB 3 – Gambar Statis
# ═══════════════════════════════════════════════════════════════
with tab_img:
    img_file = st.file_uploader("Unggah gambar",
                                type=["jpg", "jpeg", "png", "webp"])
    if img_file:
        data  = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        t0   = time.time()
        dets = detect(frame, conf_thres, iou_thres)
        ms   = (time.time() - t0) * 1000

        # Untuk gambar statis: hitung kelas dari deteksi saja (tidak ada tracker)
        img_counts = {}
        for d in dets:
            img_counts[d["cls"]] = img_counts.get(d["cls"], 0) + 1
        frame = draw_simple(frame, dets)
        frame = draw_hud(frame, img_counts, len(dets))

        col_i, col_r = st.columns([3, 1], gap="medium")
        col_i.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True)
        col_i.caption(f"⏱️ Inferensi: {ms:.0f} ms")

        with col_r:
            st.markdown("#### 📊 Hasil Deteksi")
            st.metric("Total objek", len(dets))
            if dets:
                summary = defaultdict(lambda: dict(n=0, conf_sum=0.0))
                for d in dets:
                    summary[d['cls']]['n']        += 1
                    summary[d['cls']]['conf_sum'] += d['conf']
                for cls_id in range(len(CLASS_NAMES)):
                    if cls_id in summary:
                        v = summary[cls_id]
                        hex_c = CLASS_HEX[cls_id]
                        st.markdown(
                            f'<div class="cls-card" style="background:{hex_c}cc;'
                            f'border-left:5px solid {hex_c};">'
                            f'<span>{CLASS_NAMES[cls_id]}<br>'
                            f'<small>conf {v["conf_sum"]/v["n"]:.0%}</small></span>'
                            f'<span class="cls-count">{v["n"]}</span></div>',
                            unsafe_allow_html=True)

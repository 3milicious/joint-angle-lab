"""
Joint Angle Lab  -  Streamlit app
=================================
Upload a video or static image(s) -> detect joints (MediaPipe Pose) -> segments,
joint angles, segment lengths, editable joints, reference scaling, rigid link model.
Projects are saved as Excel workbooks (.xlsx) in ./projects

Install:
    pip install streamlit mediapipe opencv-python numpy pandas openpyxl matplotlib pillow streamlit-image-coordinates
Run:
    streamlit run joint_angle_app.py

Notes
-----
* All angles are 2D (image-plane). With mixed camera views, out-of-plane motion will
  distort angles and lengths, so treat results accordingly.
* "Left"/"right" refer to the subject's left/right.
"""
import io
import json
import re
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.drawing.image import Image as XLImage
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # click-to-edit is optional; numeric editing always works
    streamlit_image_coordinates = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
PROJECT_DIR = Path("projects")
PROJECT_DIR.mkdir(exist_ok=True)
MAX_DIM = 960  # frames are resized so the longest side is at most this many pixels

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #
# joint -> MediaPipe landmark ids that are averaged to make it
JOINT_MP = {
    "head": (7, 8), "neck": (11, 12),
    "l_shoulder": (11,), "r_shoulder": (12,),
    "l_elbow": (13,), "r_elbow": (14,),
    "l_wrist": (15,), "r_wrist": (16,),
    "l_hand": (17, 19), "r_hand": (18, 20),
    "mid_hip": (23, 24), "l_hip": (23,), "r_hip": (24,),
    "l_knee": (25,), "r_knee": (26,),
    "l_ankle": (27,), "r_ankle": (28,),
    "l_heel": (29,), "r_heel": (30,),
    "l_toe": (31,), "r_toe": (32,),
}
JOINTS = list(JOINT_MP)
J = {n: i for i, n in enumerate(JOINTS)}


def build_segments():
    """name -> (proximal joint, distal joint, Winter ratio to height or None, required)."""
    s = {
        "trunk": ("mid_hip", "neck", 0.288, True),
        "head": ("neck", "head", None, True),
    }
    for sd in ("l", "r"):
        s[f"{sd}_pelvis"] = ("mid_hip", f"{sd}_hip", None, False)
        s[f"{sd}_shoulder_link"] = ("neck", f"{sd}_shoulder", None, False)
    for sd in ("l", "r"):
        s[f"{sd}_upper_arm"] = (f"{sd}_shoulder", f"{sd}_elbow", 0.186, True)
        s[f"{sd}_lower_arm"] = (f"{sd}_elbow", f"{sd}_wrist", 0.146, True)
        s[f"{sd}_hand"] = (f"{sd}_wrist", f"{sd}_hand", None, True)
        s[f"{sd}_thigh"] = (f"{sd}_hip", f"{sd}_knee", 0.245, True)
        s[f"{sd}_leg"] = (f"{sd}_knee", f"{sd}_ankle", 0.246, True)
        s[f"{sd}_foot"] = (f"{sd}_ankle", f"{sd}_toe", None, True)
        s[f"{sd}_heel_link"] = (f"{sd}_ankle", f"{sd}_heel", None, False)
    return s


SEGMENTS = build_segments()
SEG_NAMES = list(SEGMENTS)

# joint angle -> (proximal point, vertex, distal point)
ANGLES = {"neck": ("head", "neck", "mid_hip")}
for _s in ("l", "r"):
    ANGLES[f"{_s}_shoulder"] = (f"{_s}_hip", f"{_s}_shoulder", f"{_s}_elbow")
    ANGLES[f"{_s}_elbow"] = (f"{_s}_shoulder", f"{_s}_elbow", f"{_s}_wrist")
    ANGLES[f"{_s}_wrist"] = (f"{_s}_elbow", f"{_s}_wrist", f"{_s}_hand")
    ANGLES[f"{_s}_hip"] = (f"{_s}_shoulder", f"{_s}_hip", f"{_s}_knee")
    ANGLES[f"{_s}_knee"] = (f"{_s}_hip", f"{_s}_knee", f"{_s}_ankle")
    ANGLES[f"{_s}_ankle"] = (f"{_s}_knee", f"{_s}_ankle", f"{_s}_toe")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def interior(a, b, c):
    """Interior angle (deg, 0-180) at b between b->a and b->c. Works on (...,2) arrays."""
    v1, v2 = a - b, c - b
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(-1)
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def seg_len_px(coords):
    out = np.zeros((len(coords), len(SEG_NAMES)))
    for i, n in enumerate(SEG_NAMES):
        a, b, _, _ = SEGMENTS[n]
        out[:, i] = np.linalg.norm(coords[:, J[b]] - coords[:, J[a]], axis=-1)
    return out


def angles_df(p, conv):
    c = p["coords"]
    d = {"frame": np.arange(1, len(c) + 1), "time_s": np.round(p["times"], 3)}
    for n, (a, b, cc) in ANGLES.items():
        v = interior(c[:, J[a]], c[:, J[b]], c[:, J[cc]])
        d[n] = np.round(180 - v if conv == "flex" else v, 1)
    return pd.DataFrame(d)


def seg_angles_df(p):
    """Absolute segment orientation vs. horizontal (deg, counter-clockwise positive, image up = +)."""
    c = p["coords"]
    d = {"frame": np.arange(1, len(c) + 1), "time_s": np.round(p["times"], 3)}
    for n in SEG_NAMES:
        a, b, _, _ = SEGMENTS[n]
        v = c[:, J[b]] - c[:, J[a]]
        d[n] = np.round(np.degrees(np.arctan2(-v[:, 1], v[:, 0])), 1)
    return pd.DataFrame(d)


# --------------------------------------------------------------------------- #
# Reference / units
# --------------------------------------------------------------------------- #
def default_ref():
    return {"mode": "none", "height_m": None, "length_m": None,
            "joint_a": "l_shoulder", "joint_b": "l_elbow", "subject": "", "overrides": {}}


def get_scale(p):
    """metres per pixel, or None."""
    r = p["ref"]
    c = p["coords"]
    if r["mode"] == "height" and r.get("height_m"):
        px = seg_len_px(c)
        vals = []
        for i, n in enumerate(SEG_NAMES):
            ratio = SEGMENTS[n][2]
            m = np.nanmean(px[:, i])
            if ratio and np.isfinite(m) and m > 0:
                vals.append(ratio * r["height_m"] / m)
        return float(np.median(vals)) if vals else None
    if r["mode"] == "length" and r.get("length_m"):
        d = np.linalg.norm(c[:, J[r["joint_a"]]] - c[:, J[r["joint_b"]]], axis=-1)
        m = np.nanmedian(d)
        return r["length_m"] / m if np.isfinite(m) and m > 0 else None
    return None


def unit_factor(units, scale):
    """(label, multiplier from pixels to display unit)."""
    if scale is None:
        return "px", 1.0
    return ("cm", scale * 100) if units == "metric" else ("in", scale * 39.3701)


def to_m(value, units):
    return value / 100 if units == "metric" else value / 39.3701


def from_m(value_m, units):
    return value_m * 100 if units == "metric" else value_m * 39.3701


def lengths_table(p, units):
    scale = get_scale(p)
    lab, f = unit_factor(units, scale)
    px = seg_len_px(p["coords"])
    rows = []
    for i, n in enumerate(SEG_NAMES):
        mean_px, sd_px = np.nanmean(px[:, i]), np.nanstd(px[:, i])
        ov_m = p["ref"]["overrides"].get(n)
        ov = from_m(ov_m, units) if (ov_m and scale) else np.nan
        model_px = (ov_m / scale) if (ov_m and scale) else mean_px
        rows.append({
            "segment": n, "required": SEGMENTS[n][3],
            "mean_px": round(mean_px, 1), "sd_px": round(sd_px, 1),
            f"length_{lab}": round(mean_px * f, 2),
            f"sd_{lab}": round(sd_px * f, 2),
            f"override_{lab}": ov,
            f"final_{lab}": round(model_px * f, 2),
            "model_px": model_px,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Detection / media
# --------------------------------------------------------------------------- #
def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def fit_canvas(img, W, H):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((H, W, 3), 255, np.uint8)
    y0, x0 = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas


def detect(bgr, pose):
    h, w = bgr.shape[:2]
    pts, conf = np.full((len(JOINTS), 2), np.nan), np.zeros(len(JOINTS))
    res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return pts, conf
    lm = res.pose_landmarks.landmark
    P = np.array([[l.x * w, l.y * h] for l in lm])
    V = np.array([l.visibility for l in lm])
    for j, name in enumerate(JOINTS):
        ids = list(JOINT_MP[name])
        pts[j], conf[j] = P[ids].mean(0), V[ids].mean()
    return pts, conf


def process_uploads(files, stride, max_frames, progress):
    import mediapipe as mp

    tmp = Path(tempfile.mkdtemp(prefix="jal_"))
    vids = [f for f in files if f.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))]
    frames, times, fps_eff = [], [], None
    if vids:
        vp = tmp / vids[0].name
        vp.write_bytes(vids[0].getvalue())
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_eff = fps / stride
        i = 0
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(resize_max(fr))
                times.append(i / fps)
            i += 1
        cap.release()
    else:
        imgs = sorted(files, key=lambda f: f.name)
        for k, f in enumerate(imgs[:max_frames]):
            arr = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            arr = resize_max(arr)
            if frames:
                arr = fit_canvas(arr, frames[0].shape[1], frames[0].shape[0])
            frames.append(arr)
            times.append(float(k))
    if not frames:
        raise ValueError("No readable frames found in the upload.")

    H, W = frames[0].shape[:2]
    paths, coords, confs = [], [], []
    pose = mp.solutions.pose.Pose(static_image_mode=not vids, model_complexity=2 if not vids else 1,
                                  min_detection_confidence=0.5)
    for k, fr in enumerate(frames):
        pth = tmp / f"f{k:05d}.jpg"
        cv2.imwrite(str(pth), fr)
        paths.append(str(pth))
        pts, cf = detect(fr, pose)
        coords.append(pts)
        confs.append(cf)
        progress.progress((k + 1) / len(frames), text=f"Detecting joints: frame {k + 1}/{len(frames)}")
    pose.close()
    return {"coords": np.array(coords), "coords0": np.array(coords).copy(), "conf": np.array(confs),
            "times": np.array(times), "w": W, "h": H, "paths": paths, "fps_eff": fps_eff}


def get_frame(p, i):
    if p["paths"]:
        return cv2.cvtColor(cv2.imread(p["paths"][i]), cv2.COLOR_BGR2RGB)
    return np.full((p["h"], p["w"], 3), 255, np.uint8)  # reloaded workbook: no media


def joint_color(name):
    if name.startswith("l_"):
        return (30, 120, 255)
    if name.startswith("r_"):
        return (255, 140, 0)
    return (0, 190, 100)


def overlay(img, pts, sel=None, labels=False):
    out = img.copy()
    t = max(2, img.shape[1] // 320)
    for n in SEG_NAMES:
        a, b, _, _ = SEGMENTS[n]
        pa, pb = pts[J[a]], pts[J[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), joint_color(b), t)
    for j, name in enumerate(JOINTS):
        if not np.isfinite(pts[j]).all():
            continue
        c = (int(pts[j][0]), int(pts[j][1]))
        cv2.circle(out, c, t * 2, (255, 255, 255), -1)
        cv2.circle(out, c, t * 2, joint_color(name), t)
        if name == sel:
            cv2.circle(out, c, t * 5, (255, 0, 0), t)
        if labels:
            cv2.putText(out, name, (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.35, img.shape[1] / 1800), (0, 0, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Rigid link model
# --------------------------------------------------------------------------- #
def rigid_model(frame_pts, lengths_px):
    pos = {"mid_hip": frame_pts[J["mid_hip"]].copy()}
    for n in SEG_NAMES:  # ordered parent-before-child
        a, b, _, _ = SEGMENTS[n]
        d = frame_pts[J[b]] - frame_pts[J[a]]
        norm = np.hypot(*d)
        u = d / norm if norm > 0 else np.zeros(2)
        pos[b] = pos[a] + u * lengths_px[n]
    return pos


def rigid_figure(p, fi, units, show_measured, show_angles, frame_lengths, conv):
    scale = get_scale(p)
    lab, f = unit_factor(units, scale)
    lt = lengths_table(p, units)
    L = dict(zip(lt["segment"], lt["model_px"]))
    if frame_lengths:
        row = seg_len_px(p["coords"][fi:fi + 1])[0]
        L = dict(zip(SEG_NAMES, row))
    pos = rigid_model(p["coords"][fi], L)
    fig, ax = plt.subplots(figsize=(5.5, 7))
    if show_measured:
        for n in SEG_NAMES:
            a, b, _, _ = SEGMENTS[n]
            pa, pb = p["coords"][fi][J[a]] * f, p["coords"][fi][J[b]] * f
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="lightgray", lw=2, zorder=1)
    for n in SEG_NAMES:
        a, b, _, _ = SEGMENTS[n]
        pa, pb = pos[a] * f, pos[b] * f
        col = "#1e78ff" if "l_" in n else "#ff8c00" if "r_" in n else "#00be64"
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, lw=3, zorder=2)
    xy = np.array([pos[j] * f for j in JOINTS])
    ax.scatter(xy[:, 0], xy[:, 1], s=28, c="k", zorder=3)
    if show_angles:
        for n, (a, b, c) in ANGLES.items():
            v = interior(pos[a], pos[b], pos[c])
            v = 180 - v if conv == "flex" else v
            ax.annotate(f"{v:.0f}°", pos[b] * f, textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(lab)
    ax.set_ylabel(lab)
    ax.set_title(f"Rigid link model - frame {fi + 1}")
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    links = pd.DataFrame({
        "link": SEG_NAMES,
        f"length_{lab}": [round(L[n] * f, 2) for n in SEG_NAMES],
        "orientation_deg": [round(float(np.degrees(np.arctan2(
            -(pos[SEGMENTS[n][1]][1] - pos[SEGMENTS[n][0]][1]),
            pos[SEGMENTS[n][1]][0] - pos[SEGMENTS[n][0]][0]))), 1) for n in SEG_NAMES]})
    joints = pd.DataFrame({"joint": JOINTS, f"x_{lab}": np.round(xy[:, 0], 2), f"y_{lab}": np.round(xy[:, 1], 2)})
    return fig, buf.getvalue(), links, joints


# --------------------------------------------------------------------------- #
# Workbook save / load
# --------------------------------------------------------------------------- #
def safe_name(s):
    return re.sub(r"[^\w\- ]", "_", s).strip() or "project"


def build_workbook(p, units, conv):
    buf = io.BytesIO()
    scale = get_scale(p)
    lab, _ = unit_factor(units, scale)
    lt = lengths_table(p, units).drop(columns=["model_px"])
    fi = min(p.get("freeze_idx", 0), len(p["coords"]) - 1)
    _, png, links, joints = rigid_figure(p, fi, units, True, True, False, conv)
    plt.close("all")
    F = len(p["coords"])
    long = pd.DataFrame({
        "frame": np.repeat(np.arange(F), len(JOINTS)),
        "time_s": np.repeat(p["times"], len(JOINTS)),
        "joint": JOINTS * F,
        "x": p["coords"][:, :, 0].ravel(), "y": p["coords"][:, :, 1].ravel(),
        "x_detected": p["coords0"][:, :, 0].ravel(), "y_detected": p["coords0"][:, :, 1].ravel(),
        "conf": p["conf"].ravel()})
    meta = pd.DataFrame({"key": ["name", "width", "height", "fps_eff", "freeze_idx", "ref", "units", "conv"],
                         "value": [p["name"], p["w"], p["h"], p.get("fps_eff") or "", fi,
                                   json.dumps(p["ref"]), units, conv]})
    ref_rows = pd.DataFrame({"setting": ["subject", "mode", "height_m", "length_m", "joint_a", "joint_b",
                                         "scale_m_per_px", "display_unit"],
                             "value": [p["ref"]["subject"], p["ref"]["mode"], p["ref"]["height_m"],
                                       p["ref"]["length_m"], p["ref"]["joint_a"], p["ref"]["joint_b"],
                                       scale, lab]})
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        angles_df(p, conv).to_excel(xw, sheet_name="Joint Angles", index=False)
        seg_angles_df(p).to_excel(xw, sheet_name="Segment Angles", index=False)
        lt.to_excel(xw, sheet_name="Segment Lengths", index=False)
        ref_rows.to_excel(xw, sheet_name="Reference", index=False)
        links.to_excel(xw, sheet_name="Rigid Link", index=False)
        joints.to_excel(xw, sheet_name="Rigid Link", index=False, startcol=4)
        xw.sheets["Rigid Link"].add_image(XLImage(io.BytesIO(png)), "J2")
        long.to_excel(xw, sheet_name="_Joints", index=False)
        meta.to_excel(xw, sheet_name="_Meta", index=False)
    return buf.getvalue()


def save_project(p, units, conv):
    data = build_workbook(p, units, conv)
    (PROJECT_DIR / f"{safe_name(p['name'])}.xlsx").write_bytes(data)
    return data


def load_project(path):
    meta = pd.read_excel(path, sheet_name="_Meta").set_index("key")["value"]
    long = pd.read_excel(path, sheet_name="_Joints")
    F = int(long["frame"].max()) + 1
    f_idx, j_idx = long["frame"].to_numpy(int), long["joint"].map(J).to_numpy(int)
    coords, coords0 = np.full((F, len(JOINTS), 2), np.nan), np.full((F, len(JOINTS), 2), np.nan)
    conf = np.zeros((F, len(JOINTS)))
    coords[f_idx, j_idx] = long[["x", "y"]].to_numpy()
    coords0[f_idx, j_idx] = long[["x_detected", "y_detected"]].to_numpy()
    conf[f_idx, j_idx] = long["conf"].to_numpy()
    times = long.groupby("frame")["time_s"].first().to_numpy()
    ref = default_ref()
    ref.update(json.loads(meta["ref"]))
    return {"name": str(meta["name"]), "coords": coords, "coords0": coords0, "conf": conf, "times": times,
            "w": int(meta["width"]), "h": int(meta["height"]), "paths": None,
            "fps_eff": None, "ref": ref, "freeze_idx": int(meta["freeze_idx"])}, str(meta["units"]), str(meta["conv"])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Joint Angle Lab", layout="wide")
st.session_state.setdefault("proj", None)
st.session_state.setdefault("units", "metric")
st.session_state.setdefault("conv", "interior")
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("xlsx_bytes", None)

with st.sidebar:
    st.title("Joint Angle Lab")
    st.session_state.units = "metric" if st.radio(
        "Units", ["Metric (cm)", "Imperial (in)"],
        index=0 if st.session_state.units == "metric" else 1).startswith("Metric") else "imperial"
    st.session_state.conv = "flex" if st.radio(
        "Angle convention", ["Interior angle (180° = straight)", "Flexion (180° − interior; 0° = straight)"],
        index=0 if st.session_state.conv == "interior" else 1).startswith("Flexion") else "interior"
    p = st.session_state.proj
    st.divider()
    if p:
        st.write(f"**Active project:** {p['name']}")
        if st.button("💾 Save workbook", use_container_width=True):
            st.session_state.xlsx_bytes = save_project(p, st.session_state.units, st.session_state.conv)
            st.success(f"Saved projects/{safe_name(p['name'])}.xlsx")
        if st.session_state.xlsx_bytes:
            st.download_button("⬇ Download .xlsx", st.session_state.xlsx_bytes,
                               file_name=f"{safe_name(p['name'])}.xlsx", use_container_width=True)
    else:
        st.info("No active project. Open the Files tab.")

units, conv = st.session_state.units, st.session_state.conv
tab_files, tab_an, tab_ref, tab_rigid = st.tabs(["📁 Files", "1 · Angles & Lengths", "2 · Reference", "3 · Rigid Link Model"])

# ----------------------------- FILES ---------------------------------------- #
with tab_files:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("New project")
        name = st.text_input("Project name", value="Project 1")
        files = st.file_uploader("Video or image(s)", accept_multiple_files=True,
                                 type=["mp4", "mov", "avi", "mkv", "webm", "jpg", "jpeg", "png", "bmp"])
        stride = st.slider("Video: analyse every Nth frame", 1, 30, 3)
        max_frames = st.slider("Max frames", 1, 600, 200)
        if st.button("Detect joints", type="primary", disabled=not files):
            try:
                bar = st.progress(0.0, text="Starting...")
                data = process_uploads(files, stride, max_frames, bar)
                data.update({"name": name, "ref": default_ref(), "freeze_idx": 0})
                st.session_state.proj = data
                st.session_state.xlsx_bytes = save_project(data, units, conv)
                st.success("Done. Project created and saved. Go to tab 1.")
                st.rerun()
            except ImportError:
                st.error("MediaPipe is not installed: pip install mediapipe")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with c2:
        st.subheader("Saved workbooks")
        saved = sorted(PROJECT_DIR.glob("*.xlsx"))
        if saved:
            pick = st.selectbox("Workbook", saved, format_func=lambda x: x.stem)
            b1, b2 = st.columns(2)
            if b1.button("Open"):
                proj, u, cv = load_project(pick)
                st.session_state.proj, st.session_state.units, st.session_state.conv = proj, u, cv
                st.session_state.xlsx_bytes = None
                st.rerun()
            if b2.button("Delete"):
                pick.unlink()
                st.rerun()
            st.caption("Reopened workbooks keep all joint edits and settings but not the video; "
                       "joints are drawn on a blank canvas.")
        else:
            st.caption("No saved workbooks yet.")

# ----------------------------- ANALYSIS ------------------------------------- #
p = st.session_state.proj
if not p:
    for t in (tab_an, tab_ref, tab_rigid):
        with t:
            st.info("Create or open a project in the Files tab first.")
    st.stop()

F = len(p["coords"])

with tab_an:
    fi = st.slider("Frame", 1, F, 1) - 1 if F > 1 else 0
    left, right = st.columns([3, 2])
    with right:
        st.markdown("**Edit joints**")
        sel = st.selectbox("Joint", JOINTS)
        click_mode = st.checkbox("Click on image to move selected joint",
                                 disabled=streamlit_image_coordinates is None)
        if streamlit_image_coordinates is None:
            st.caption("Install `streamlit-image-coordinates` for click-to-edit.")
        cur = p["coords"][fi, J[sel]]
        cur = np.nan_to_num(cur, nan=0.0)
        x = st.number_input("x (px)", value=float(cur[0]), format="%.1f")
        y = st.number_input("y (px)", value=float(cur[1]), format="%.1f")
        e1, e2, e3 = st.columns(3)
        if e1.button("Apply"):
            p["coords"][fi, J[sel]] = [x, y]
            st.rerun()
        if e2.button("Reset joint"):
            p["coords"][fi, J[sel]] = p["coords0"][fi, J[sel]]
            st.rerun()
        if e3.button("Reset frame"):
            p["coords"][fi] = p["coords0"][fi]
            st.rerun()
        n_edit = int(np.sum(np.nan_to_num(p["coords"] - p["coords0"], nan=1.0) != 0) // 2)
        st.caption(f"{n_edit} edited joint positions in project. Angles, lengths and the rigid model "
                   "update automatically.")
        labels = st.checkbox("Show joint labels", value=False)

    with left:
        ov = overlay(get_frame(p, fi), p["coords"][fi], sel, labels)
        if streamlit_image_coordinates is not None:
            click = streamlit_image_coordinates(Image.fromarray(ov), key=f"img_{fi}", width=min(p["w"], 800))
            if click and click_mode and click.get("unix_time") != st.session_state.last_click:
                st.session_state.last_click = click.get("unix_time")
                p["coords"][fi, J[sel]] = [click["x"] * p["w"] / click["width"],
                                           click["y"] * p["h"] / click["height"]]
                st.rerun()
        else:
            st.image(ov)

    adf = angles_df(p, conv)
    st.subheader("Joint angles (°)")
    st.dataframe(adf, height=300)

    st.subheader("Estimated segment lengths")
    lt_view = lengths_table(p, units).drop(columns=["model_px", "required"])
    st.dataframe(lt_view)
    if get_scale(p) is None:
        st.caption("No reference set: lengths are in pixels. Add one in tab 2 to get cm / inches.")

    with st.expander("Segment orientation angles (vs. horizontal)"):
        st.dataframe(seg_angles_df(p))

    st.subheader("Real-time angle view")
    if st.toggle("Enable real-time view"):
        cols = st.multiselect("Angles to plot", list(ANGLES), default=["l_knee", "r_knee"])
        speed = st.select_slider("Playback speed", [0.25, 0.5, 1.0, 2.0], value=1.0)
        if cols:
            st.line_chart(adf.set_index("time_s")[cols])
        play = st.button("▶ Play from current frame")
        ph_img, ph_tbl = st.empty(), st.empty()
        if play:
            for i in range(fi, F):
                ph_img.image(overlay(get_frame(p, i), p["coords"][i]), caption=f"Frame {i + 1}/{F}",
                             width=min(p["w"], 640))
                ph_tbl.dataframe(adf.iloc[[i]][["frame", "time_s"] + (cols or list(ANGLES))])
                dt = (p["times"][i + 1] - p["times"][i]) if i + 1 < F else 0
                time.sleep(max(0.02, min(dt / speed, 1.0)))

# ----------------------------- REFERENCE ------------------------------------ #
with tab_ref:
    r = p["ref"]
    u_lab = "cm" if units == "metric" else "in"
    st.subheader("Reference measurement")
    r["subject"] = st.text_input("Subject label (optional)", value=r["subject"])
    modes = {"none": "None (pixels only; angles are unaffected)",
             "height": "Known person height",
             "length": "Known distance between two joints"}
    r["mode"] = st.radio("Reference type", list(modes), format_func=modes.get,
                         index=list(modes).index(r["mode"]))
    if r["mode"] == "height":
        h_in = st.number_input(f"Height ({u_lab})", min_value=0.0,
                               value=float(from_m(r["height_m"], units)) if r["height_m"] else 0.0)
        r["height_m"] = to_m(h_in, units) if h_in > 0 else None
        st.caption("Scale is estimated from limb/trunk proportions (Winter anthropometry). "
                   "Best when the subject is roughly perpendicular to the camera.")
    elif r["mode"] == "length":
        a, b = st.columns(2)
        r["joint_a"] = a.selectbox("Joint A", JOINTS, index=JOINTS.index(r["joint_a"]))
        r["joint_b"] = b.selectbox("Joint B", JOINTS, index=JOINTS.index(r["joint_b"]))
        l_in = st.number_input(f"Known distance A-B ({u_lab})", min_value=0.0,
                               value=float(from_m(r["length_m"], units)) if r["length_m"] else 0.0)
        r["length_m"] = to_m(l_in, units) if l_in > 0 else None
    scale = get_scale(p)
    st.write("**Scale:** " + (f"{scale * 100:.4f} cm/px" if scale else "not set (pixel units)"))

    st.subheader("Modify segment lengths")
    if scale:
        lt = lengths_table(p, units)
        view = lt[["segment", f"length_{u_lab}", f"override_{u_lab}"]].rename(
            columns={f"length_{u_lab}": f"estimated_{u_lab}"})
        edited = st.data_editor(view, disabled=["segment", f"estimated_{u_lab}"], hide_index=True,
                                key="ov_editor")
        if st.button("Apply overrides"):
            r["overrides"] = {row["segment"]: to_m(row[f"override_{u_lab}"], units)
                              for _, row in edited.iterrows() if pd.notna(row[f"override_{u_lab}"])}
            st.rerun()
        st.caption("Overrides feed the rigid link model and the 'final' length column. "
                   "Clear a cell and apply to go back to the measured value.")
    else:
        st.info("Set a reference above to enable length overrides.")

# ----------------------------- RIGID LINK ----------------------------------- #
with tab_rigid:
    st.subheader("Rigid link model from freeze frame")
    fz = st.slider("Freeze frame", 1, F, min(p.get("freeze_idx", 0) + 1, F), key="freeze") - 1 if F > 1 else 0
    p["freeze_idx"] = fz
    o1, o2, o3 = st.columns(3)
    show_meas = o1.checkbox("Overlay measured pose", True)
    show_ang = o2.checkbox("Show joint angles", True)
    frame_len = o3.checkbox("Use this frame's lengths (otherwise project average / overrides)", False)
    fig, png, links, joints = rigid_figure(p, fz, units, show_meas, show_ang, frame_len, conv)
    cA, cB = st.columns([2, 3])
    with cA:
        st.pyplot(fig)
        st.download_button("⬇ Download PNG", png, file_name=f"rigid_link_frame{fz + 1}.png")
    with cB:
        st.markdown("**Links**")
        st.dataframe(links)
        st.markdown("**Joint coordinates**")
        st.dataframe(joints)
    plt.close(fig)

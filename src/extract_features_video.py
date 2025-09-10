#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze fitness clip (cross-view, exercise-agnostic) with noise-robust segmentation.

- Downsample a TARGET_FPS (default 15 fps)
- Extract 2D + 3D world landmarks (MediaPipe) via PoseEstimator (import from src.keypoints_extractor)
- Kalman smoothing on 3D
- Body-centric transform + essential 3D, view-agnostic features
- Robust view estimation (front/semi_front/side/semi_back/back) + smoothing
- Noise-robust ACTIVE SEGMENT detection (pre/post movement removed)
- Rep detection (hysteresis, durations, refractory) ONLY inside active segments
- Save keyframe overlays (optional)
- Output JSON with features, time-series, segments, and reps

Run:
    python analyze_clip.py --meta data/benchmark/scraped_json/BWSQUAT.json \
                           --out output_crossview_verbose.json \
                           --keyframes-dir kf_out --verbose
"""
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

from src.keypoints_extractor import PoseEstimator  # <-- tua classe

# =========================
# Config principali
# =========================
TARGET_FPS = 15          # fps analisi: meno latenza/CPU, sufficiente per movimenti umani
VIS_TH = 0.5             # soglia visibilità landmark (MediaPipe visibility)
SMOOTH_WIN = 7           # smoothing temporale "morbido" (frames) su serie
# Kalman 1D modello posizione-velocità
KALMAN_Q = 1e-3
KALMAN_R = 4e-3

# Segmentazione attività (rumore pre/post)
WIN_SEC = 3.0            # finestra scorrevole per statistiche (s)
WARMUP_SEC = 2.0         # durata usata per baseline se all'inizio c'è quiete
SEG_MIN_ACTIVE_S = 1.2   # durata minima di un segmento attivo
SEG_MERGE_GAP_S = 0.35   # gap massimo tra due cluster da unire
SEG_THR_STD_GAIN = 2.0   # moltiplicatore su baseline std per attivazione
SEG_THR_ROM_GAIN = 1.2   # moltiplicatore su baseline ROM per attivazione

# Rep detection (robusta)
MIN_REL_ROM = 0.14
MIN_REP_DURATION_S   = 0.50
MIN_PHASE_DURATION_S = 0.18
MIN_GAP_BETWEEN_REPS_S = 0.20

# =========================
# Indici MediaPipe Pose
# =========================
class POSE:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_MOUTH = 9
    RIGHT_MOUTH = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# =========================
# Helpers geometrici
# =========================
def _xy_from_landmark(lmk, frame_shape):
    h, w = frame_shape[:2]
    return np.array([lmk.x*w, lmk.y*h], dtype=np.float32)

def angle3p_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab, cb = a - b, c - b
    denom = (np.linalg.norm(ab)*np.linalg.norm(cb) + 1e-9)
    return float(math.degrees(math.acos(np.clip(np.dot(ab, cb)/denom, -1.0, 1.0))))

def smooth_signal(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x
    k = min(win, len(x))
    if k % 2 == 0:
        k -= 1
    if k < 1:
        return x
    kernel = np.ones(k, dtype=np.float32)/k
    return np.convolve(x, kernel, mode='same')

def vprint(verbose: bool, *args):
    if verbose:
        print(*args)

# =========================
# IO helpers
# =========================
def load_meta_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def iter_video_frames(path: str, target_fps: int = TARGET_FPS):
    """
    Itera sul video restituendo (frame_bgr, t_sec, idx) downsamplati a target_fps.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    eff_fps = src_fps / step

    raw_idx = 0
    out_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_idx % step == 0:
            yield frame, float(out_idx/eff_fps), int(out_idx)
            out_idx += 1
        raw_idx += 1
    cap.release()

def presence_score(kp2d: Dict[int, Dict[str,float]]) -> float:
    """Percentuale landmark con visibilità >= VIS_TH."""
    if not kp2d:
        return 0.0
    v = [d.get('v',0.0) for d in kp2d.values()]
    ok = sum(1 for z in v if z >= VIS_TH)
    return ok / max(len(v),1)

# =========================
# Estrazione 2D + 3D
# =========================
def extract_pose_sequence(video_path: str, pose_estimator, vis_th: float = VIS_TH, verbose: bool = True):
    """
    Ritorna frames con:
      - 'keypoints' 2D in px (solo se visibility >= vis_th)
      - 'world' 3D in "metri" MediaPipe (con v copiato da 2D)
      - 'presence' [0..1]
    """
    frames, shapes = [], []
    total = 0
    for frame_bgr, t_sec, idx in iter_video_frames(video_path, TARGET_FPS):
        total += 1
        res = pose_estimator.process(frame_bgr)
        shapes.append(frame_bgr.shape)

        kp2d = {}
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                if lm.visibility >= vis_th:
                    x, y = _xy_from_landmark(lm, frame_bgr.shape)
                    kp2d[i] = {'x': float(x), 'y': float(y), 'v': float(lm.visibility)}

        kp3d = {}
        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                v = kp2d.get(i, {}).get('v', 0.0)
                kp3d[i] = {'X': float(lm.x), 'Y': float(lm.y), 'Z': float(lm.z), 'v': float(v)}

        presence = presence_score(kp2d)
        frames.append({
            'frame_idx': idx,
            't': float(t_sec),
            'keypoints': kp2d,
            'world': kp3d,
            'presence': presence
        })
        if verbose and idx % 30 == 0:
            print(f"[extract] frame={idx:4d} t={t_sec:5.2f}s  visible%={presence*100:4.1f}  kp2d={len(kp2d)}  kp3d={len(kp3d)}")

    if verbose:
        print(f"[extract] tot_frame_analizzati={total}  (downsample a ~{TARGET_FPS} fps)")
    return frames, shapes

# =========================
# Kalman smoothing 3D
# =========================
class Kalman1D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self.x = np.zeros((2,1), dtype=np.float32)  # [pos, vel]
        self.P = np.eye(2, dtype=np.float32)
        self.q = float(q)
        self.r = float(r)
        self.initialized = False

    def predict(self, dt):
        F = np.array([[1, dt],[0,1]], dtype=np.float32)
        Q = np.array([[self.q*dt*dt, 0],[0, self.q]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1,0]], dtype=np.float32)
        R = np.array([[self.r]], dtype=np.float32)
        y = np.array([[z]], dtype=np.float32) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, z, dt):
        if not self.initialized:
            self.x = np.array([[z],[0]], dtype=np.float32)
            self.P = np.eye(2, dtype=np.float32)
            self.initialized = True
            return float(z)
        self.predict(dt)
        self.update(z)
        return float(self.x[0,0])

def kalman_smooth_world(frames: List[Dict[str,Any]], joints: List[int] = list(range(33))) -> List[Dict[int, Dict[str,float]]]:
    filters = {(j,ax): Kalman1D() for j in joints for ax in 'XYZ'}
    out = []
    prev_t = frames[0]['t'] if frames else 0.0
    for f in frames:
        dt = max(f['t'] - prev_t, 1.0/TARGET_FPS)
        prev_t = f['t']
        sm = {}
        for j in joints:
            if j in f['world']:
                X, Y, Z = f['world'][j]['X'], f['world'][j]['Y'], f['world'][j]['Z']
                v = f['world'][j]['v']
                Xs = filters[(j,'X')].step(X, dt)
                Ys = filters[(j,'Y')].step(Y, dt)
                Zs = filters[(j,'Z')].step(Z, dt)
                sm[j] = {'X':Xs, 'Y':Ys, 'Z':Zs, 'v':v}
        out.append(sm)
    return out

# =========================
# Body-centric frame + View robust
# =========================
def body_frame_from_world(world_pts: Dict[int, Dict[str,float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    need = [POSE.LEFT_HIP, POSE.RIGHT_HIP, POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER]
    if any(i not in world_pts for i in need):
        return None
    LHIP = np.array([world_pts[POSE.LEFT_HIP][k]  for k in 'XYZ'], dtype=np.float32)
    RHIP = np.array([world_pts[POSE.RIGHT_HIP][k] for k in 'XYZ'], dtype=np.float32)
    LSH  = np.array([world_pts[POSE.LEFT_SHOULDER][k]  for k in 'XYZ'], dtype=np.float32)
    RSH  = np.array([world_pts[POSE.RIGHT_SHOULDER][k] for k in 'XYZ'], dtype=np.float32)

    hip_mid = 0.5*(LHIP + RHIP)
    sh_mid  = 0.5*(LSH + RSH)

    x_axis = RHIP - LHIP; x_axis /= (np.linalg.norm(x_axis)+1e-9)
    y_axis = sh_mid - hip_mid; y_axis /= (np.linalg.norm(y_axis)+1e-9)
    z_axis = np.cross(x_axis, y_axis); z_axis /= (np.linalg.norm(z_axis)+1e-9)
    y_axis = np.cross(z_axis, x_axis); y_axis /= (np.linalg.norm(y_axis)+1e-9)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows = body axes in world basis
    origin = hip_mid
    return R, origin

def to_body_coords(world_pts: Dict[int, Dict[str,float]]) -> Optional[Dict[int, np.ndarray]]:
    bf = body_frame_from_world(world_pts)
    if bf is None:
        return None
    R, origin = bf
    out = {}
    for i, v in world_pts.items():
        P = np.array([v['X'], v['Y'], v['Z']], dtype=np.float32)
        out[i] = R @ (P - origin)  # world->body
    return out

def _yaw_from_shoulders_3d(world_pts: Dict[int, Dict[str,float]]) -> Optional[float]:
    need = [POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER]
    if any(i not in world_pts for i in need):
        return None
    L = world_pts[POSE.LEFT_SHOULDER]; R = world_pts[POSE.RIGHT_SHOULDER]
    dx = float(R['X'] - L['X'])
    dz = float(R['Z'] - L['Z'])
    return float(math.degrees(math.atan2(dz, dx + 1e-9)))

def _face_visibility_score(kp2d: Dict[int, Dict[str,float]]) -> float:
    ids = [POSE.NOSE, POSE.LEFT_EYE, POSE.RIGHT_EYE]
    return float(sum(kp2d.get(i, {}).get('v', 0.0) for i in ids))

def estimate_view_robust(world_pts: Dict[int, Dict[str,float]],
                         kp2d: Optional[Dict[int, Dict[str,float]]] = None) -> Dict[str, Optional[float]]:
    need = [POSE.LEFT_HIP, POSE.RIGHT_HIP, POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER]
    if any(i not in world_pts for i in need):
        return {'view':'unknown','yaw_deg':None,'roll_deg':None,'pitch_deg':None,'front_score':None}

    Rinfo = body_frame_from_world(world_pts)
    if Rinfo is None:
        return {'view':'unknown','yaw_deg':None,'roll_deg':None,'pitch_deg':None,'front_score':None}
    R, _ = Rinfo
    x_body, y_body, z_body = R[0], R[1], R[2]
    roll  = math.degrees(math.atan2(x_body[1], np.linalg.norm([x_body[0], x_body[2]]) + 1e-9))
    pitch = math.degrees(math.atan2(y_body[2], y_body[1] + 1e-9))

    yaw = _yaw_from_shoulders_3d(world_pts)
    if yaw is None:
        return {'view':'unknown','yaw_deg':None,'roll_deg':float(roll),'pitch_deg':float(pitch),'front_score':None}
    yaw_abs = abs(yaw)

    face_vis = _face_visibility_score(kp2d or {})
    front_score = min(face_vis / 1.5, 1.0)  # ~[0..1]

    T_SEMI = 25.0
    T_SIDE = 65.0
    if yaw_abs <= T_SEMI:
        if front_score >= 0.6:
            view = 'front'
        elif front_score >= 0.35:
            view = 'semi_front'
        else:
            view = 'back'
    elif yaw_abs <= T_SIDE:
        if front_score >= 0.5:
            view = 'semi_front'
        elif front_score <= 0.2:
            view = 'semi_back'
        else:
            view = 'side'
    else:
        view = 'side' if front_score >= 0.25 else 'semi_back'

    return {'view':view, 'yaw_deg':float(yaw), 'roll_deg':float(roll), 'pitch_deg':float(pitch), 'front_score':float(front_score)}

# =========================
# Feature essenziali 3D
# =========================
ESSENTIAL_JOINT_ANGLES_3D = {
    'shoulder_L_deg_3d': (POSE.LEFT_ELBOW,  POSE.LEFT_SHOULDER,  POSE.LEFT_HIP),
    'shoulder_R_deg_3d': (POSE.RIGHT_ELBOW, POSE.RIGHT_SHOULDER, POSE.RIGHT_HIP),
    'elbow_L_deg_3d':    (POSE.LEFT_SHOULDER,  POSE.LEFT_ELBOW,  POSE.LEFT_WRIST),
    'elbow_R_deg_3d':    (POSE.RIGHT_SHOULDER, POSE.RIGHT_ELBOW, POSE.RIGHT_WRIST),
    'hip_L_deg_3d':      (POSE.LEFT_SHOULDER,  POSE.LEFT_HIP,  POSE.LEFT_KNEE),
    'hip_R_deg_3d':      (POSE.RIGHT_SHOULDER, POSE.RIGHT_HIP, POSE.RIGHT_KNEE),
    'knee_L_deg_3d':     (POSE.LEFT_HIP,  POSE.LEFT_KNEE,  POSE.LEFT_ANKLE),
    'knee_R_deg_3d':     (POSE.RIGHT_HIP, POSE.RIGHT_KNEE, POSE.RIGHT_ANKLE),
    'ankle_L_deg_3d':    (POSE.LEFT_KNEE,  POSE.LEFT_ANKLE,  POSE.LEFT_FOOT_INDEX),
    'ankle_R_deg_3d':    (POSE.RIGHT_KNEE, POSE.RIGHT_ANKLE, POSE.RIGHT_FOOT_INDEX),
}

def estimate_view_sequence(frames: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Calcola view per frame (robusta) e la smussa temporalmente (yaw + majority sui label).
    """
    from collections import deque, Counter
    views_raw, yaws = [], []
    for f in frames:
        vi = estimate_view_robust(f.get('world', {}), kp2d=f.get('keypoints', {}))
        views_raw.append(vi)
        yaws.append(vi.get('yaw_deg'))

    # smoothing yaw
    buf = deque(maxlen=9)
    for vi, y in zip(views_raw, yaws):
        if y is not None:
            buf.append(y)
            vi['yaw_deg'] = float(np.mean(buf))

    # majority vote su label
    buf_lab = deque(maxlen=9)
    for vi in views_raw:
        lab = vi['view']
        if lab != 'unknown':
            buf_lab.append(lab)
        if len(buf_lab) > 0:
            from collections import Counter
            vi['view'] = Counter(buf_lab).most_common(1)[0][0]
    return views_raw

def compute_essential_features(frames: List[Dict[str,Any]],
                               world_smoothed: List[Dict[int, Dict[str,float]]],
                               user_hint: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    # prepara input per view smoothing
    frames_for_view = []
    for f, w3 in zip(frames, world_smoothed):
        frames_for_view.append({
            'world': w3,
            'keypoints': f.get('keypoints', {}),
            't': f['t'],
            'frame_idx': f['frame_idx'],
            'presence': f['presence'],
        })
    views_seq = estimate_view_sequence(frames_for_view)

    out = []
    for f, w3, vinfo in zip(frames, world_smoothed, views_seq):
        q = {'visible_ratio': float(f['presence'])}
        body = to_body_coords(w3)

        feats = {}
        # Angoli 3D invarianti
        if w3:
            def get3(i):
                if i in w3:
                    return np.array([w3[i]['X'], w3[i]['Y'], w3[i]['Z']], dtype=np.float32)
                return None
            for name, (a,b,c) in ESSENTIAL_JOINT_ANGLES_3D.items():
                pa, pb, pc = get3(a), get3(b), get3(c)
                feats[name] = angle3p_3d(pa,pb,pc) if (pa is not None and pb is not None and pc is not None) else None

        # Distanze normalizzate nel body frame
        if body is not None and all(k in body for k in [POSE.LEFT_ANKLE, POSE.RIGHT_ANKLE, POSE.LEFT_HIP, POSE.RIGHT_HIP,
                                                        POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER]):
            ankle_L, ankle_R = body[POSE.LEFT_ANKLE], body[POSE.RIGHT_ANKLE]
            hip_L, hip_R     = body[POSE.LEFT_HIP],   body[POSE.RIGHT_HIP]
            sh_L, sh_R       = body[POSE.LEFT_SHOULDER], body[POSE.RIGHT_SHOULDER]
            hip_w = np.linalg.norm(hip_R - hip_L) + 1e-6
            stance = np.linalg.norm(ankle_R - ankle_L)
            feats['stance_width_norm'] = float(stance / hip_w)
            hip_mid = 0.5*(hip_L + hip_R)
            sh_mid  = 0.5*(sh_L + sh_R)
            feats['pelvis_y_body'] = float(hip_mid[1])
            v = sh_mid - hip_mid
            y_axis = np.array([0,1,0], dtype=np.float32)
            denom = (np.linalg.norm(v)*np.linalg.norm(y_axis)+1e-9)
            feats['torso_incline_deg'] = float(math.degrees(math.acos(np.clip(np.dot(v,y_axis)/denom, -1,1))))
        else:
            feats['stance_width_norm'] = None
            feats['pelvis_y_body'] = None
            feats['torso_incline_deg'] = None

        if user_hint and isinstance(user_hint.get('height_cm'), (int,float)):
            feats['user_height_cm'] = float(user_hint['height_cm'])

        out.append({
            'frame_idx': f['frame_idx'],
            't': f['t'],
            'presence': f['presence'],
            'quality': q,
            'view': vinfo,
            'features_3d': feats
        })
    return out

# =========================
# Time-series + quality
# =========================
ESSENTIAL_KEYS_TS = [
    'shoulder_L_deg_3d','shoulder_R_deg_3d',
    'elbow_L_deg_3d','elbow_R_deg_3d',
    'hip_L_deg_3d','hip_R_deg_3d',
    'knee_L_deg_3d','knee_R_deg_3d',
    'ankle_L_deg_3d','ankle_R_deg_3d',
    'stance_width_norm','pelvis_y_body','torso_incline_deg'
]

def extract_series(frames_out: List[Dict[str,Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([f['t'] for f in frames_out], dtype=np.float32)
    x = []
    for f in frames_out:
        val = f['features_3d'].get(key)
        x.append(np.nan if val is None else float(val))
    return t, np.array(x, dtype=np.float32)

def clip_quality(frames_out: List[Dict[str,Any]]) -> Dict[str, float]:
    vis = np.array([f['presence'] for f in frames_out], dtype=np.float32) if frames_out else np.array([])
    return {
        'mean_presence': float(np.mean(vis)) if vis.size else 0.0,
        'min_presence': float(np.min(vis)) if vis.size else 0.0,
        'frames': int(len(frames_out))
    }

def series_package(frames_out: List[Dict[str,Any]]) -> Dict[str, Any]:
    pkg = {}
    for k in ESSENTIAL_KEYS_TS:
        t, x = extract_series(frames_out, k)
        xs = smooth_signal(x, SMOOTH_WIN)
        pkg[k] = {'t': t.tolist(), 'series': xs.tolist(),
                  'rom': (float(np.nanmax(xs)-np.nanmin(xs)) if xs.size else None)}
    return pkg

# =========================
# Rep detection robusta
# =========================
@dataclass
class Rep:
    start_idx: int
    bottom_idx: int
    end_idx: int
    start_t: float
    bottom_t: float
    end_t: float
    signal_key: str
    prom_min: float
    rom: float
    duration: float

def _find_local_extrema(x: np.ndarray) -> Tuple[List[int], List[int]]:
    maxima, minima = [], []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            maxima.append(i)
        if x[i] < x[i-1] and x[i] < x[i+1]:
            minima.append(i)
    return maxima, minima

def _detect_reps_on_signal(t: np.ndarray, x: np.ndarray, key: str) -> List[Rep]:
    if len(x) < 5:
        return []
    xs = smooth_signal(x.copy(), SMOOTH_WIN)
    maxima, minima = _find_local_extrema(xs)
    if not maxima or not minima:
        return []
    glob_rom = float(np.nanmax(xs) - np.nanmin(xs))
    thr_prom = float(MIN_REL_ROM * max(glob_rom, 1e-6))
    # velocità
    v = np.zeros_like(xs)
    for i in range(1,len(xs)-1):
        dt = max(t[i+1]-t[i-1], 1e-6)
        v[i] = (xs[i+1]-xs[i-1])/dt
    v[0] = (xs[1]-xs[0])/max(t[1]-t[0],1e-6)
    v[-1]= (xs[-1]-xs[-2])/max(t[-1]-t[-2],1e-6)

    reps = []
    last_end_t = -1e9
    last_end_idx = -10**9
    for mi in minima:
        tops_before = [j for j in maxima if j < mi]
        tops_after  = [j for j in maxima if j > mi]
        if not tops_before or not tops_after:
            continue
        s = tops_before[-1]; e = tops_after[0]
        # refrattario
        if s <= last_end_idx or (t[s]-last_end_t) < MIN_GAP_BETWEEN_REPS_S:
            continue
        # prominence
        prom_left  = float(xs[s] - xs[mi])
        prom_right = float(xs[e] - xs[mi])
        prom_min = min(prom_left, prom_right)
        if prom_left < thr_prom or prom_right < thr_prom:
            continue
        # durate
        t_ecc = float(t[mi]-t[s]); t_con = float(t[e]-t[mi]); t_rep = float(t[e]-t[s])
        if t_rep < MIN_REP_DURATION_S or t_ecc < MIN_PHASE_DURATION_S or t_con < MIN_PHASE_DURATION_S:
            continue
        # segno velocità coerente
        if np.median(v[s:mi+1]) >= 0:  # deve scendere
            continue
        if np.median(v[mi:e+1]) <= 0:  # deve salire
            continue
        rep_rom = float(max(xs[s], xs[e]) - xs[mi])
        reps.append(Rep(s, mi, e, float(t[s]), float(t[mi]), float(t[e]), key, float(prom_min), float(rep_rom), float(t_rep)))
        last_end_t = t[e]; last_end_idx = e
    return reps

def detect_reps_essential(frames_out: List[Dict[str,Any]], verbose: bool = True) -> Tuple[List[Rep], str]:
    """
    Sceglie il miglior segnale tra:
      - 'pelvis_y_body' (invertito)
      - angoli 3D (knee/hip/shoulder/elbow/ankle)
    """
    candidates = [
        ('pelvis_y_body', True),
        ('knee_L_deg_3d', False), ('knee_R_deg_3d', False),
        ('hip_L_deg_3d', False),  ('hip_R_deg_3d', False),
        ('shoulder_L_deg_3d', False), ('shoulder_R_deg_3d', False),
        ('elbow_L_deg_3d', False),    ('elbow_R_deg_3d', False),
        ('ankle_L_deg_3d', False),    ('ankle_R_deg_3d', False),
    ]
    t = np.array([f['t'] for f in frames_out], dtype=np.float32)
    best_reps, best_key, best_score = [], None, (-1, -np.inf, -np.inf)

    for key, invert in candidates:
        x = np.array([ (f['features_3d'].get(key) if f['features_3d'].get(key) is not None else np.nan)
                       for f in frames_out ], dtype=np.float32)
        if np.all(np.isnan(x)):
            continue
        xw = -x if invert else x
        reps = _detect_reps_on_signal(t, xw, key)
        if not reps:
            continue
        rep_count = len(reps)
        med_prom  = float(np.median([r.prom_min for r in reps]))
        med_rom   = float(np.median([r.rom for r in reps]))
        score = (rep_count, med_prom, med_rom)
        if score > best_score:
            best_score = score; best_reps = reps; best_key = key

    if verbose:
        if best_key:
            print(f"[reps] segnale='{best_key}'  reps={len(best_reps)}  med_prom={best_score[1]:.2f}  med_rom={best_score[2]:.2f}")
        else:
            print("[reps] nessuna ripetizione rilevata con i segnali essenziali")

    # etichetta i keyframe a bordo del frames_out
    for r in best_reps:
        def set_if_none(i, tag):
            frames_out[i]['label'] = frames_out[i].get('label') or tag
        set_if_none(r.start_idx, 'top_start')
        set_if_none(r.bottom_idx,'bottom')
        set_if_none(r.end_idx,  'top_end')
        mid_ecc = (r.start_idx + r.bottom_idx)//2
        mid_con = (r.bottom_idx + r.end_idx)//2
        set_if_none(mid_ecc, 'mid_eccentric')
        set_if_none(mid_con, 'mid_concentric')

    return best_reps, (best_key or 'none')

# =========================
# Segmentazione attività (rumore-robust)
# =========================
def active_segments(frames_out: List[Dict[str,Any]],
                    primary_key: str = 'pelvis_y_body',
                    fps: int = TARGET_FPS) -> List[Tuple[int,int]]:
    """
    Trova intervalli [i0, i1] (inclusivi) dove il movimento è “attivo”:
      - usa sliding window WIN_SEC sul segnale principale (pelvis_y_body se presente, altrimenti ROM max tra chiavi)
      - calcola baseline (std/ROM) su inizio clip (WARMUP_SEC) o sulla finestra minima
      - attivo se std > gain*baseline_std OR ROM > gain*baseline_rom
      - unisce gap piccoli, scarta segmenti corti
    Ritorna lista di indici frame (start_idx, end_idx).
    """
    n = len(frames_out)
    if n == 0:
        return []

    # scegli segnale
    def get_series_for_key(k):
        return np.array([ (f['features_3d'].get(k) if f['features_3d'].get(k) is not None else np.nan)
                          for f in frames_out ], dtype=np.float32)
    x = get_series_for_key(primary_key)
    if np.all(np.isnan(x)):
        # fallback: scegli chiave con ROM più alta
        roms = {}
        for k in ['pelvis_y_body','knee_L_deg_3d','knee_R_deg_3d','hip_L_deg_3d','hip_R_deg_3d']:
            xi = get_series_for_key(k)
            if np.all(np.isnan(xi)): continue
            xs = smooth_signal(xi, SMOOTH_WIN)
            roms[k] = float(np.nanmax(xs)-np.nanmin(xs))
        if not roms:
            return []
        primary_key = max(roms.items(), key=lambda kv: kv[1])[0]
        x = get_series_for_key(primary_key)

    # fill NaN con riempimenti bordi
    def fill_nan(a):
        b = a.copy()
        # ffill
        for i in range(1,len(b)):
            if np.isnan(b[i]) and not np.isnan(b[i-1]): b[i] = b[i-1]
        # bfill
        for i in range(len(b)-2,-1,-1):
            if np.isnan(b[i]) and not np.isnan(b[i+1]): b[i] = b[i+1]
        return b
    x = fill_nan(x)
    xs = smooth_signal(x, SMOOTH_WIN)

    win = max(3, int(WIN_SEC*fps))
    # baseline: se inizio clip è fermo, usalo; altrimenti prendi la finestra con std minima
    def rolling_std_rom(sig):
        stds, roms = [], []
        for i in range(n):
            j0 = max(0, i-win+1)
            seg = sig[j0:i+1]
            seg = seg[~np.isnan(seg)]
            if seg.size < 3:
                stds.append(np.nan); roms.append(np.nan); continue
            stds.append(float(np.std(seg)))
            roms.append(float(np.max(seg)-np.min(seg)))
        return np.array(stds), np.array(roms)
    stds, roms = rolling_std_rom(xs)

    warm = min(n, int(WARMUP_SEC*fps))
    if warm >= 5:
        base_std = float(np.nanmean(stds[:warm]))
        base_rom = float(np.nanmean(roms[:warm]))
    else:
        # minima finestra
        base_std = float(np.nanmin(stds[np.isfinite(stds)])) if np.isfinite(stds).any() else 0.0
        base_rom = float(np.nanmin(roms[np.isfinite(roms)])) if np.isfinite(roms).any() else 0.0

    thr_std = base_std * SEG_THR_STD_GAIN + 1e-6
    thr_rom = base_rom * SEG_THR_ROM_GAIN + 1e-6

    active_mask = (stds > thr_std) | (roms > thr_rom)
    # dilata/merge piccoli buchi
    gap = int(max(1, round(SEG_MERGE_GAP_S*fps)))
    # un semplice fill di piccoli zeri tra uno e l'altro
    i = 0
    while i < n:
        if not active_mask[i]:
            # cerca lunghezza buco
            j = i
            while j < n and not active_mask[j]:
                j += 1
            # se tra due blocchi attivi e il buco è corto => fill
            if i > 0 and j < n and (j - i) <= gap:
                active_mask[i:j] = True
            i = j
        else:
            i += 1

    # estrai segmenti e filtra per durata
    segs = []
    i = 0
    while i < n:
        if active_mask[i]:
            j = i
            while j < n and active_mask[j]:
                j += 1
            if (j - i) / fps >= SEG_MIN_ACTIVE_S:
                segs.append((i, j-1))
            i = j
        else:
            i += 1
    return segs

# =========================
# Visualizzazione keyframe
# =========================
def extract_keyframe_indices_from_reps(reps: List[Rep]) -> List[int]:
    idxs = []
    for r in reps:
        idxs += [r.start_idx, (r.start_idx+r.bottom_idx)//2, r.bottom_idx, (r.bottom_idx+r.end_idx)//2, r.end_idx]
    return sorted(set(idxs))

def save_keyframe_overlays(video_path: str,
                           frames_out: List[Dict[str,Any]],
                           key_indices: List[int],
                           pose_estimator,
                           out_dir: str = "debug_keyframes",
                           visibility_th: float = VIS_TH,
                           verbose: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    labels = {f['frame_idx']: (f.get('label') or '') for f in frames_out if f.get('label')}
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / TARGET_FPS)))
    raw_idx = 0
    out_idx = 0
    sel = set(key_indices)
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if raw_idx % step == 0:
            if out_idx in sel:
                res = pose_estimator.process(frame)
                overlay = pose_estimator.draw_landmarks(frame, res.pose_landmarks if res else None, visibility_th=visibility_th)
                tag = labels.get(out_idx, '')
                txt = f"{out_idx:04d}" + (f" | {tag}" if tag else "")
                cv2.putText(overlay, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
                out_path = os.path.join(out_dir, f"kf_{out_idx:04d}.jpg")
                cv2.imwrite(out_path, overlay)
                saved += 1
            out_idx += 1
        raw_idx += 1
    cap.release()
    if verbose:
        print(f"[viz] keyframes salvati: {saved} in '{out_dir}'")

# =========================
# Pipeline end-to-end (robusta al rumore)
# =========================
def process_clip_crossview(meta_in: Dict[str,Any],
                           pose_estimator,
                           out_json_path: Optional[str] = None,
                           user_hint: Optional[Dict[str,Any]] = None,
                           save_keyframes: bool = True,
                           keyframes_dir: str = "debug_keyframes",
                           verbose: bool = True) -> Dict[str,Any]:

    if 'media' not in meta_in or 'clip_path' not in meta_in['media']:
        raise ValueError("meta_in deve contenere media.clip_path")
    video_path = meta_in['media']['clip_path']
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    vprint(verbose, f"[start] file='{video_path}'  TARGET_FPS={TARGET_FPS}")

    # 1) Pose 2D+3D
    frames_raw, _ = extract_pose_sequence(video_path, pose_estimator, vis_th=VIS_TH, verbose=verbose)
    vprint(verbose, f"[presence] media visibilità: {np.mean([f['presence'] for f in frames_raw]):.3f}")

    # 2) Kalman su 3D
    vprint(verbose, "[kalman] smoothing world landmarks ...")
    world_smoothed = kalman_smooth_world(frames_raw)

    # 3) Feature 3D essenziali + view + quality
    vprint(verbose, "[features] calcolo feature 3D essenziali ...")
    frames_out = compute_essential_features(frames_raw, world_smoothed, user_hint=user_hint)

    # 4) Segmentazione attività (rumore robusto)
    vprint(verbose, "[segment] ricerca segmenti attivi ...")
    segs = active_segments(frames_out, primary_key='pelvis_y_body', fps=TARGET_FPS)
    if verbose:
        if segs:
            print("[segment] attivi:", ", ".join([f"[{a}-{b}]" for a,b in segs]))
        else:
            print("[segment] nessun segmento attivo robusto trovato (potrebbe essere tutto rumore)")

    # 5) Rep detection: SOLO nei segmenti attivi
    reps_all: List[Rep] = []
    used_signal = None
    for (a,b) in segs if segs else [(0,len(frames_out)-1)]:
        sub = frames_out[a:b+1]
        reps, used_signal = detect_reps_essential(sub, verbose=verbose)
        # riallinea indici al clip completo
        for r in reps:
            reps_all.append(Rep(
                start_idx=r.start_idx + a,
                bottom_idx=r.bottom_idx + a,
                end_idx=r.end_idx + a,
                start_t=sub[r.start_idx]['t'],
                bottom_t=sub[r.bottom_idx]['t'],
                end_t=sub[r.end_idx]['t'],
                signal_key=r.signal_key,
                prom_min=r.prom_min,
                rom=r.rom,
                duration=r.duration
            ))
    # etichetta keyframes globali
    for r in reps_all:
        def set_if_none(i, tag):
            frames_out[i]['label'] = frames_out[i].get('label') or tag
        set_if_none(r.start_idx, 'top_start')
        set_if_none(r.bottom_idx,'bottom')
        set_if_none(r.end_idx,  'top_end')
        mid_ecc = (r.start_idx + r.bottom_idx)//2
        mid_con = (r.bottom_idx + r.end_idx)//2
        set_if_none(mid_ecc, 'mid_eccentric')
        set_if_none(mid_con, 'mid_concentric')

    kf_idxs = extract_keyframe_indices_from_reps(reps_all)
    vprint(verbose, f"[reps] conteggio ripetizioni = {len(reps_all)}  | keyframes estratti = {len(kf_idxs)}")

    # 6) Time series + qualità clip
    vprint(verbose, "[timeseries] estrazione serie e ROM ...")
    ts = series_package(frames_out)
    qclip = clip_quality(frames_out)

    # 7) View dominante
    views = [f['view'].get('view') for f in frames_out if f.get('view')]
    dom_view = None
    if views:
        vals, counts = np.unique([v or 'unknown' for v in views], return_counts=True)
        dom_view = str(vals[np.argmax(counts)])
    if user_hint and user_hint.get('view_hint'):
        dom_view = user_hint['view_hint']
    vprint(verbose, f"[view] dominante: {dom_view}")

    # 8) Salva keyframes
    if save_keyframes and len(kf_idxs) > 0:
        vprint(verbose, "[viz] salvataggio keyframes con overlay ...")
        save_keyframe_overlays(video_path, frames_out, kf_idxs, pose_estimator, out_dir=keyframes_dir, visibility_th=VIS_TH, verbose=verbose)

    # 9) Output
    out_obj = dict(meta_in)
    out_obj.setdefault('analysis', {})
    out_obj['analysis'].update({
        'pose': {'fps_effective': TARGET_FPS, 'dominant_view': dom_view},
        'clip_quality': qclip,
        'segments_active': segs,
        'rep_detection': {
            'signal_used': used_signal,
            'rep_count': len(reps_all),
            'keyframes_indices': kf_idxs
        },
        'frame_features': frames_out,
        'time_series': ts
    })

    if out_json_path is not None:
        save_json(out_obj, out_json_path)
        vprint(verbose, f"[done] output salvato in: {out_json_path}")

    return out_obj

# =========================
# main
# =========================
# =========================
# main (tutti i parametri in chiaro)
# =========================
if __name__ == "__main__":
    # ---- PARAMETRI "IN CHIARO" PER PROVARE ----
    META_PATH       = "data/benchmark/scraped_json/BWSQUAT.json"  # JSON d’ingresso con media.clip_path
    OUT_JSON_PATH   = "output_crossview_verbose.json"              # dove salvare l'output (None per non salvare)
    USER_HEIGHT_CM  = None                                   # None se non vuoi passare l'altezza
    VIEW_HINT       = None  # "front" | "side" | "back" | "semi_front" | "semi_back" | None
    SAVE_KEYFRAMES  = True
    KEYFRAMES_DIR   = "kf_out"
    MODEL_COMPLEXITY = 1   # 0 | 1 | 2
    VERBOSE         = True

    current_directory = os.getcwd()
    print("La directory corrente è:", current_directory)
    parent_directory = os.path.dirname(current_directory)
    print("La directory genitore è:", parent_directory)
    os.chdir(parent_directory)
    print("La nuova directory corrente è:", os.getcwd())

    # ---- CARICA META + INIZIALIZZA POSE ----
    meta = load_meta_json(META_PATH)
    pose_est = PoseEstimator(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False
    )

    # ---- HINT OPZIONALI UTENTE ----
    user_hint = {}
    if USER_HEIGHT_CM is not None:
        user_hint['height_cm'] = float(USER_HEIGHT_CM)
    if VIEW_HINT is not None:
        user_hint['view_hint'] = str(VIEW_HINT)

    # ---- ESECUZIONE PIPELINE ----
    out = process_clip_crossview(
        meta_in = meta,
        pose_estimator = pose_est,
        out_json_path = OUT_JSON_PATH,
        user_hint = user_hint if user_hint else None,
        save_keyframes = SAVE_KEYFRAMES,
        keyframes_dir = KEYFRAMES_DIR,
        verbose = VERBOSE
    )

    # ---- SUMMARY ----
    rep_count = out['analysis']['rep_detection']['rep_count']
    dom_view  = out['analysis']['pose']['dominant_view']
    frames_n  = out['analysis']['clip_quality']['frames']
    print(f"\n[summary] reps={rep_count}  view={dom_view}  frames={frames_n}")
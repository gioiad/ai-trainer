# --- Cell 1: Config & imports ---
import os, time, json, math, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
import requests
import mediapipe as mp

from src.keypoints_extractor import PoseEstimator
from src.features_extractor import PoseFeatureExtractor

import cv2, numpy as np, os, requests

def safe_imread(path_or_url:str, fallback_size=(480,640)):
    """
    - Se path_or_url è un file esistente -> lo carica.
    - Altrimenti tenta di scaricarlo come URL e salvarlo in 'test_input.jpg'.
    - Se fallisce, ritorna un frame nero (placeholder).
    """
    img = None
    if os.path.exists(path_or_url):
        img = cv2.imread(path_or_url)
    else:
        try:
            r = requests.get(path_or_url, timeout=10)
            r.raise_for_status()
            with open("test_input.jpg", "wb") as f:
                f.write(r.content)
            img = cv2.imread("test_input.jpg")
        except Exception:
            img = None

    if img is None:
        img = np.zeros((fallback_size[0], fallback_size[1], 3), dtype=np.uint8)
        print("⚠️ Immagine non trovata/non scaricata: uso placeholder nero.")
    return img

if __name__ == "__main__":

    # Ollama
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

    DISPLAY_SCALE = 0.9

    user_profile = {
        "name": "Alex",
        "age": 32,
        "sex": "M",
        "height_cm": 178,
        "weight_kg": 76,
        "level": "intermediate",
        "goals": ["ricomposizione corporea", "forza", "resistenza"],
        "preferences": {
            "tone": "motivational_friendly",
            "cue_style": "visual_plus_short_verbal",
            "language": "it-IT"
        },
        "equipment": {"dumbbells": True, "mini_barbell": True, "bands": True, "mat": True},
        "constraints": {"lower_back_sensitivity": False, "shoulder_limitations": False, "knee_limitations": False}
    }

    img = safe_imread("../data/image/squat.jpg")

    # Richiede: PoseEstimator definita nelle celle precedenti
    pe = PoseEstimator(static_image_mode=True)

    frame = img  # usa l'immagine caricata con safe_imread
    res = pe.process(frame)

    if res and res.pose_landmarks:
        out = pe.draw_landmarks(frame, res.pose_landmarks)
        cv2.imwrite("keypoints_preview.jpg", out)
        print("✅ Keypoints disegnati. File: keypoints_preview.jpg")
        print("Landmarks rilevati:", len(res.pose_landmarks.landmark))  # atteso 33
    else:
        print("❌ Nessun corpo rilevato: prova un'altra foto (intera, ben illuminata).")

    PSF = PoseFeatureExtractor()

    # Richiede: compute_features(...) definita nelle celle precedenti
    if res and res.pose_landmarks:
        feats = PSF.compute_features(res.pose_landmarks, frame.shape)
        # stampa ordinata e leggibile
        nice = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in feats.items()}
        print("📊 Features:", nice)
    else:
        print("❌ Niente features: nessun landmark trovato.")


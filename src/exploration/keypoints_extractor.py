import os, time, json, math, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
import requests
import mediapipe as mp

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)

    def draw_landmarks(self, frame_bgr, landmarks, visibility_th=0.5, highlight: Dict[int, Tuple[int,int,int]]=None):
        if landmarks is None:
            return frame_bgr
        image = frame_bgr.copy()
        self.drawing.draw_landmarks(
            image, landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
        )
        if highlight:
            h, w = image.shape[:2]
            for idx, color in highlight.items():
                lmk = landmarks.landmark[idx]
                if lmk.visibility >= visibility_th:
                    cx, cy = int(lmk.x*w), int(lmk.y*h)
                    cv2.circle(image, (cx, cy), 8, color, -1)
        return image


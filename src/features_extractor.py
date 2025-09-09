from typing import Dict, Tuple
import math
import numpy as np
import mediapipe as mp


class PoseFeatureExtractor:
    """
    Estrattore di feature basato su MediaPipe Pose.

    Inizializzazione
    ---------------
    POSE : oggetto enum di MediaPipe (es. mp.solutions.pose.PoseLandmark)
    valgus_threshold : soglia (in unità normalizzate) per il flag di valgismo del ginocchio
    """

    def __init__(self,  valgus_threshold: float = 0.08):
        self.POSE = mp.solutions.pose.PoseLandmark
        self.valgus_threshold = float(valgus_threshold)

    # -------------------------
    # Utility geometriche base
    # -------------------------
    def _get_xy(self, landmarks, idx, frame_shape) -> Tuple[np.ndarray, float]:
        h, w = frame_shape[:2]
        l = landmarks.landmark[idx]
        return np.array([l.x * w, l.y * h], dtype=np.float32), float(l.visibility)

    def angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Restituisce l'angolo ABC in gradi."""
        ab = a - b
        cb = c - b
        denom = (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
        cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
        return float(math.degrees(math.acos(cosang)))

    def line_point_distance(self, p1: np.ndarray, p2: np.ndarray, p: np.ndarray) -> float:
        """Distanza del punto p dalla retta p1-p2 (in pixel)."""
        num = np.abs(np.cross(p2 - p1, p1 - p))
        den = (np.linalg.norm(p2 - p1) + 1e-6)
        return float(num / den)

    # -------------------------
    # Angoli articolari
    # -------------------------
    def knee_angle(self, landmarks, side: str, frame_shape) -> float:
        hip, knee, ankle = (
            (self.POSE.LEFT_HIP, self.POSE.LEFT_KNEE, self.POSE.LEFT_ANKLE)
            if side == "left"
            else (self.POSE.RIGHT_HIP, self.POSE.RIGHT_KNEE, self.POSE.RIGHT_ANKLE)
        )
        A, _ = self._get_xy(landmarks, hip, frame_shape)
        B, _ = self._get_xy(landmarks, knee, frame_shape)
        C, _ = self._get_xy(landmarks, ankle, frame_shape)
        return self.angle(A, B, C)

    def hip_angle(self, landmarks, side: str, frame_shape) -> float:
        shoulder, hip, knee = (
            (self.POSE.LEFT_SHOULDER, self.POSE.LEFT_HIP, self.POSE.LEFT_KNEE)
            if side == "left"
            else (self.POSE.RIGHT_SHOULDER, self.POSE.RIGHT_HIP, self.POSE.RIGHT_KNEE)
        )
        A, _ = self._get_xy(landmarks, shoulder, frame_shape)
        B, _ = self._get_xy(landmarks, hip, frame_shape)
        C, _ = self._get_xy(landmarks, knee, frame_shape)
        return self.angle(A, B, C)

    def elbow_angle(self, landmarks, side: str, frame_shape) -> float:
        shoulder, elbow, wrist = (
            (self.POSE.LEFT_SHOULDER, self.POSE.LEFT_ELBOW, self.POSE.LEFT_WRIST)
            if side == "left"
            else (self.POSE.RIGHT_SHOULDER, self.POSE.RIGHT_ELBOW, self.POSE.RIGHT_WRIST)
        )
        A, _ = self._get_xy(landmarks, shoulder, frame_shape)
        B, _ = self._get_xy(landmarks, elbow, frame_shape)
        C, _ = self._get_xy(landmarks, wrist, frame_shape)
        return self.angle(A, B, C)

    def shoulder_angle(self, landmarks, side: str, frame_shape) -> float:
        """Angolo busto–spalla–gomito."""
        hip, shoulder, elbow_j = (
            (self.POSE.LEFT_HIP, self.POSE.LEFT_SHOULDER, self.POSE.LEFT_ELBOW)
            if side == "left"
            else (self.POSE.RIGHT_HIP, self.POSE.RIGHT_SHOULDER, self.POSE.RIGHT_ELBOW)
        )
        A, _ = self._get_xy(landmarks, hip, frame_shape)
        B, _ = self._get_xy(landmarks, shoulder, frame_shape)
        C, _ = self._get_xy(landmarks, elbow_j, frame_shape)
        return self.angle(A, B, C)

    def ankle_angle(self, landmarks, side: str, frame_shape) -> float:
        """Angolo ginocchio–caviglia–punta."""
        knee_j, ankle_j, foot_j = (
            (self.POSE.LEFT_KNEE, self.POSE.LEFT_ANKLE, self.POSE.LEFT_FOOT_INDEX)
            if side == "left"
            else (self.POSE.RIGHT_KNEE, self.POSE.RIGHT_ANKLE, self.POSE.RIGHT_FOOT_INDEX)
        )
        A, _ = self._get_xy(landmarks, knee_j, frame_shape)
        B, _ = self._get_xy(landmarks, ankle_j, frame_shape)
        C, _ = self._get_xy(landmarks, foot_j, frame_shape)
        return self.angle(A, B, C)

    # -------------------------
    # Altre metriche
    # -------------------------
    def pelvis_drop_norm(self, landmarks, frame_shape) -> float:
        Lhip, _ = self._get_xy(landmarks, self.POSE.LEFT_HIP, frame_shape)
        Rhip, _ = self._get_xy(landmarks, self.POSE.RIGHT_HIP, frame_shape)
        Lsh, _ = self._get_xy(landmarks, self.POSE.LEFT_SHOULDER, frame_shape)
        Lank, _ = self._get_xy(landmarks, self.POSE.LEFT_ANKLE, frame_shape)
        body_len = np.linalg.norm(Lsh - Lank) + 1e-6
        return float((Lhip[1] - Rhip[1]) / body_len)  # >0 se anca sinistra più bassa

    # -------------------------
    # Feature principali
    # -------------------------
    def compute_features(self, landmarks, frame_shape) -> Dict[str, float]:
        if landmarks is None:
            keys = [
                "knee_ang","hip_ang","elbow_ang","shoulder_ang","ankle_ang",
                "depth","valgus_flag","knee_valgus_score",
                "left_knee_over_ankle","right_knee_over_ankle",
                "torso_tilt","pelvis_drop_norm","shoulder_hip_ankle_collinearity",
                "arm_symmetry","leg_symmetry","stance_width","grip_width",
                "bar_vertical_disp","shoulderY","wristY",
                "knee_to_chest_norm_L","knee_to_chest_norm_R",
                "crossbody_knee_wrist_prox_L","crossbody_knee_wrist_prox_R",
                "hands_to_feet_min_dist","tuck_compactness","arms_overhead",
            ]
            return {k: 0.0 for k in keys}

        h, w = frame_shape[:2]

        # Angoli medi
        knee = (self.knee_angle(landmarks, "left", frame_shape) +
                self.knee_angle(landmarks, "right", frame_shape)) / 2
        hipA = (self.hip_angle(landmarks, "left", frame_shape) +
                self.hip_angle(landmarks, "right", frame_shape)) / 2
        elbow = (self.elbow_angle(landmarks, "left", frame_shape) +
                 self.elbow_angle(landmarks, "right", frame_shape)) / 2
        shoulderA = (self.shoulder_angle(landmarks, "left", frame_shape) +
                     self.shoulder_angle(landmarks, "right", frame_shape)) / 2
        ankleA = (self.ankle_angle(landmarks, "left", frame_shape) +
                  self.ankle_angle(landmarks, "right", frame_shape)) / 2

        # Punti chiave
        Lhip, _ = self._get_xy(landmarks, self.POSE.LEFT_HIP, frame_shape)
        Rhip, _ = self._get_xy(landmarks, self.POSE.RIGHT_HIP, frame_shape)
        Lkne, _ = self._get_xy(landmarks, self.POSE.LEFT_KNEE, frame_shape)
        Rkne, _ = self._get_xy(landmarks, self.POSE.RIGHT_KNEE, frame_shape)
        Lank, _ = self._get_xy(landmarks, self.POSE.LEFT_ANKLE, frame_shape)
        Rank, _ = self._get_xy(landmarks, self.POSE.RIGHT_ANKLE, frame_shape)
        Lwri, _ = self._get_xy(landmarks, self.POSE.LEFT_WRIST, frame_shape)
        Rwri, _ = self._get_xy(landmarks, self.POSE.RIGHT_WRIST, frame_shape)
        Lsho, _ = self._get_xy(landmarks, self.POSE.LEFT_SHOULDER, frame_shape)
        Rsho, _ = self._get_xy(landmarks, self.POSE.RIGHT_SHOULDER, frame_shape)

        mid_sho = (Lsho + Rsho) / 2
        mid_hip = (Lhip + Rhip) / 2
        mid_ank = (Lank + Rank) / 2

        # Profondità squat
        depth = float(((Lhip[1] + Rhip[1]) / 2) - ((Lkne[1] + Rkne[1]) / 2))

        # Valgismo normalizzato
        hip_width = abs(Rhip[0] - Lhip[0]) + 1e-6
        left_kot = (Lkne[0] - Lank[0]) / hip_width
        right_kot = (Rkne[0] - Rank[0]) / hip_width
        knee_valgus_score = max(0.0, -left_kot) + max(0.0, right_kot)
        valgus_flag = 1 if knee_valgus_score > self.valgus_threshold else 0

        # Torso / collinearità (plank)
        torso_vec = mid_sho - mid_hip
        torso_tilt = math.degrees(math.atan2(torso_vec[0], torso_vec[1]))  # 0 = verticale
        sha = mid_sho
        ank = mid_ank
        hip_line_dist = self.line_point_distance(sha, ank, mid_hip)
        norm_len = (np.linalg.norm(sha - ank) + 1e-6)
        shoulder_hip_ankle_collinearity = float(hip_line_dist / norm_len)

        # Simmetrie & larghezze
        arm_symmetry = float(
            self.elbow_angle(landmarks, "left", frame_shape)
            - self.elbow_angle(landmarks, "right", frame_shape)
        )
        leg_symmetry = float(
            self.knee_angle(landmarks, "left", frame_shape)
            - self.knee_angle(landmarks, "right", frame_shape)
        )
        stance_width = float(abs(Lank[0] - Rank[0]))
        grip_width = float(abs(Lwri[0] - Rwri[0]))

        # ROM verticale (press/trazioni)
        shoulderY = float((Lsho[1] + Rsho[1]) / 2)
        wristY = float((Lwri[1] + Rwri[1]) / 2)
        bar_vertical_disp = float(shoulderY - wristY)

        # Mountain climber: knee->chest & cross-body
        leg_len = float(np.linalg.norm(mid_hip - mid_ank) + 1e-6)
        knee_to_chest_norm_L = float(np.linalg.norm(Lkne - mid_sho) / leg_len)
        knee_to_chest_norm_R = float(np.linalg.norm(Rkne - mid_sho) / leg_len)
        crossbody_knee_wrist_prox_L = float(np.linalg.norm(Lkne - Rwri) / (norm_len + 1e-6))
        crossbody_knee_wrist_prox_R = float(np.linalg.norm(Rkne - Lwri) / (norm_len + 1e-6))

        # Burpees: mani <-> piedi, tuck, braccia sopra la testa
        hands_to_feet_min_dist = float(
            min(
                np.linalg.norm(Lwri - Lank),
                np.linalg.norm(Lwri - Rank),
                np.linalg.norm(Rwri - Lank),
                np.linalg.norm(Rwri - Rank),
            )
            / (norm_len + 1e-6)
        )
        tuck_compactness = float(min(knee_to_chest_norm_L, knee_to_chest_norm_R))
        arms_overhead = 1.0 if wristY < shoulderY - 10 else 0.0  # 10 px di margine

        return {
            "knee_ang": float(knee),
            "hip_ang": float(hipA),
            "elbow_ang": float(elbow),
            "shoulder_ang": float(shoulderA),
            "ankle_ang": float(ankleA),
            "depth": depth,
            "valgus_flag": int(valgus_flag),
            "knee_valgus_score": float(knee_valgus_score),
            "left_knee_over_ankle": float(left_kot),
            "right_knee_over_ankle": float(right_kot),
            "torso_tilt": float(torso_tilt),
            "pelvis_drop_norm": float(self.pelvis_drop_norm(landmarks, frame_shape)),
            "shoulder_hip_ankle_collinearity": float(shoulder_hip_ankle_collinearity),
            "arm_symmetry": arm_symmetry,
            "leg_symmetry": leg_symmetry,
            "stance_width": stance_width,
            "grip_width": grip_width,
            "bar_vertical_disp": bar_vertical_disp,
            "shoulderY": shoulderY,
            "wristY": wristY,
            "knee_to_chest_norm_L": knee_to_chest_norm_L,
            "knee_to_chest_norm_R": knee_to_chest_norm_R,
            "crossbody_knee_wrist_prox_L": crossbody_knee_wrist_prox_L,
            "crossbody_knee_wrist_prox_R": crossbody_knee_wrist_prox_R,
            "hands_to_feet_min_dist": hands_to_feet_min_dist,
            "tuck_compactness": tuck_compactness,
            "arms_overhead": arms_overhead,
        }


if __name__ == "__main__":
    # Esempio d'uso minimale (place-holder):
    # from mediapipe import solutions as mp_solutions
    # extractor = PoseFeatureExtractor(mp_solutions.pose.PoseLandmark)
    # features = extractor.compute_features(landmarks, frame.shape)
    # print(features)
    pass

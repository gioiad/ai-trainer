import json
import os
import numpy as np
import requests
import cv2
import time
from pathlib import Path
from typing import Tuple, List, Optional, Any, Dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webcam runner: ascolta costantemente dalla webcam, segmenta clip per "set"
usando un semplice motion detector, e per ogni set invoca process_video_crossview
per ottenere il rep_assessment che viene salvato come JSON in data/test/webcam_feed/.

Nota: i video non vengono salvati, solo i JSON.
"""

# importa dalla tua pipeline esistente
from extract_features_video import (
    process_video_crossview,
    PoseEstimator,
    TARGET_FPS
)


# ----------------------------
# Utility
# ----------------------------
def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_llm_prompt_from_template(template_path: str,
                                   benchmark_obj: Dict[str, Any],
                                   candidate_obj: Dict[str, Any]) -> str:
    """
    Carica il template testuale e inserisce i due JSON (stringati) nei placeholder.
    """
    template = Path(template_path).read_text(encoding="utf-8")

    bench_str = json.dumps(benchmark_obj, ensure_ascii=False, separators=(",", ":"))
    cand_str  = json.dumps(candidate_obj, ensure_ascii=False, separators=(",", ":"))

    prompt = (template
              .replace("<<BENCHMARK_JSON>>", bench_str)
              .replace("<<CANDIDATE_JSON>>", cand_str))
    return prompt

def call_ollama_gemma3(prompt: str,
                       model: str = "gemma3",
                       url: str = "http://localhost:11434/api/generate",
                       stream: bool = False,
                       options: Dict[str, Any] = None) -> str:
    """
    Chiama l'API locale di Ollama e ritorna la risposta testuale dell’LLM.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    if options:
        payload["options"] = options

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # per /api/generate, il testo è in data["response"]
    return data.get("response", "")

# ----------------------------
# Webcam loop
# ----------------------------
# ----------------------------
# Webcam loop
# ----------------------------
def run_webcam_session():
    import time
    import json
    import requests
    from pathlib import Path

    # ==== PARAMETRI IN CHIARO ====
    cam_index = 0
    out_dir = "data/test/webcam_feed"
    target_fps = TARGET_FPS
    idle_end_s = 2.0       # tempo di inattività per chiudere un set
    min_clip_s = 1.5       # durata minima di un set valido
    max_clip_s = 30.0      # durata massima di un set
    motion_resize = (160,120)
    motion_alpha = 0.12  # <— prima era 0.05: EMA più reattiva
    motion_thr = 3.0  # <— prima era 7.0: soglia più bassa
    show = True            # mostra preview webcam
    coach_mode = "reps"    # "gt" | "reps" | "last"
    save_keyframes = False # non necessario in live

    calib_secs = 2.0
    calib_vals = []
    calib_start = None

    # ==== PARAMETRI LLM / PROMPT IN CHIARO ====
    benchmark_json_path  = "output_rep_assessment_gt.json"  # <-- GT già generato
    prompt_template_path = "src/prompt/prompt_llm_coach_squat.txt"            # <-- deve contenere i placeholder:
                                                                         #     <<BENCHMARK_JSON>> e <<CANDIDATE_JSON>>
    ollama_model   = "gemma3"
    ollama_url     = "http://localhost:11434/api/generate"
    ollama_stream  = False
    ollama_options = None  # es. {"temperature": 0.2}

    # ---- setup I/O live ----
    out_path = ensure_dir(out_dir)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam index={cam_index}")

    pose_est = PoseEstimator(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False
    )

    dt_target = 1.0 / max(1, target_fps)
    prev_small = None
    ema = 0.0

    state = "idle"
    last_active_t: Optional[float] = None
    last_written_t = 0.0

    buf_frames: List[Any] = []
    clip_start_t: Optional[float] = None

    print(f"[live] webcam index={cam_index}  target_fps={target_fps}")
    print("[live] premi 'q' per uscire")

    # ---- helper locali per LLM ----
    def _build_llm_prompt(template_path: str, benchmark_obj: dict, candidate_obj: dict) -> str:
        template = Path(template_path).read_text(encoding="utf-8")
        bench_str = json.dumps(benchmark_obj, ensure_ascii=False, separators=(",", ":"))
        cand_str  = json.dumps(candidate_obj, ensure_ascii=False, separators=(",", ":"))
        return (template
                .replace("<<BENCHMARK_JSON>>", bench_str)
                .replace("<<CANDIDATE_JSON>>", cand_str))

    def _call_ollama(prompt_text: str) -> str:
        payload = {
            "model": ollama_model,
            "prompt": prompt_text,
            "stream": ollama_stream
        }
        if ollama_options:
            payload["options"] = ollama_options
        r = requests.post(ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()

    try:
        calib_start = time.monotonic()
        # --- HighGUI: crea e posiziona la finestra prima del loop
        window_name = "webcam_live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # finestra ridimensionabile
        cv2.resizeWindow(window_name, 960, 540)  # dimensione iniziale
        cv2.moveWindow(window_name, 80, 80)  # portala in primo piano/visibile
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[live] frame non letto (stream terminato?) — esco")
                break

            # --- Motion score ---
            small = cv2.resize(frame, motion_resize)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_small is None:
                prev_small = gray.copy()
            diff = cv2.absdiff(gray, prev_small)
            prev_small = gray
            score = float(diff.mean())
            ema = (1.0 - motion_alpha) * ema + motion_alpha * score
            now = time.monotonic()
            if (now - calib_start) <= calib_secs:
                calib_vals.append(ema)
            elif calib_vals:
                base = float(np.mean(calib_vals))
                spread = float(np.std(calib_vals))
                motion_thr = base + 2.5 * max(spread, 0.4)  # fattore 2.5 è un buon punto di partenza
                calib_vals = []  # disattiva calibrazione dopo il primo set
            active = (ema >= motion_thr)

            # --- FSM ACTIVE/IDLE ---
            if active:
                last_active_t = now
                if state == "idle":
                    state = "active"
                    buf_frames.clear()
                    clip_start_t = now
                    last_written_t = 0.0
                    print("[live] >>> ACTIVE (inizio set)")
            else:
                if state == "active" and last_active_t and (now - last_active_t) >= idle_end_s:
                    state = "idle"
                    clip_len_s = (len(buf_frames) / float(target_fps)) if target_fps > 0 else 0.0
                    print(f"[live] <<< IDLE (fine set) frames={len(buf_frames)} ~{clip_len_s:.2f}s")

                    if clip_len_s >= min_clip_s:
                        # --- processiamo i frame del set (senza salvare video persistente) ---
                        ts = time.strftime("%Y%m%d_%H%M%S")

                        # MP4 temporaneo (poi cancellato) per riusare la pipeline esistente
                        tmp_path = str(Path(out_path) / f"_tmp_{ts}.mp4")
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(tmp_path, fourcc, target_fps, (w, h))
                        for fr in buf_frames:
                            writer.write(fr)
                        writer.release()

                        # esegui estrazione rep_assessment
                        rep_assessment = process_video_crossview(
                            video_path=tmp_path,
                            pose_estimator=pose_est,
                            user_hint=None,
                            save_keyframes=save_keyframes,
                            keyframes_dir=str(out_path / "kf_out"),
                            verbose=False,
                            coach_mode=coach_mode
                        )

                        # cancella il temporaneo
                        Path(tmp_path).unlink(missing_ok=True)

                        # se non sono state trovate reps, non ha senso chiamare l'LLM
                        rc = rep_assessment.get("payload", {}).get("reps_count", 0)
                        if rc and rc > 0:
                            try:
                                # carica benchmark GT e costruisci il prompt
                                with open(benchmark_json_path, "r", encoding="utf-8") as f:
                                    benchmark_obj = json.load(f)
                                candidate_obj = rep_assessment
                                prompt_text = _build_llm_prompt(prompt_template_path, benchmark_obj, candidate_obj)

                                # chiama Ollama e stampa feedback
                                llm_reply = _call_ollama(prompt_text)
                                print("\n================ LLM FEEDBACK ================\n")
                                print(llm_reply)
                                print("\n==============================================\n")
                            except Exception as e:
                                print(f"[live] errore LLM: {e}")
                        else:
                            print("[live] nessuna ripetizione valida trovata in questo set — salto LLM")

                    else:
                        print("[live] set troppo corto: scarto")

                    buf_frames.clear()
                    clip_start_t = None

            # --- Accumulo frame ---
            if state == "active":
                if (now - last_written_t) >= dt_target:
                    buf_frames.append(frame.copy())
                    last_written_t = now
                    if clip_start_t and (now - clip_start_t) >= max_clip_s:
                        print("[live] max_clip_s raggiunto → chiusura forzata")
                        last_active_t = now - idle_end_s - 1.0  # forza uscita a breve

            # --- Preview ---
            if show:
                vis = frame.copy()
                cv2.putText(
                    vis,
                    f"state:{state}  ema:{ema:.1f} thr:{motion_thr:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if state == "active" else (0, 0, 255),
                    2
                )
                try:
                    cv2.imshow(window_name, vis)
                except cv2.error as e:
                    print("[live][HighGUI] errore imshow:", e)
                    print("Probabile build headless. Installa 'opencv-python' non headless.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[live] quit richiesto")
                    break
                elif key == ord('+'):
                    motion_thr += 0.5
                    print(f"[live] motion_thr -> {motion_thr:.2f}")
                elif key == ord('-'):
                    motion_thr = max(0.0, motion_thr - 0.5)
                    print(f"[live] motion_thr -> {motion_thr:.2f}")

    except KeyboardInterrupt:
        print("\n[live] interrotto dall'utente")

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        print("[live] chiuso")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    current_directory = os.getcwd()
    print("La directory corrente è:", current_directory)
    parent_directory = os.path.dirname(current_directory)
    print("La directory genitore è:", parent_directory)
    os.chdir(parent_directory)
    print("La nuova directory corrente è:", os.getcwd())
    run_webcam_session()
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LogicVisualizer:
    def __init__(self):
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Data stores
        self.calib_data = []  # (timestamp, EAR)
        self.perclos_data = {
            "time": [],
            "ear": [],
            "is_closed": [],
            "perclos_val": []
        }
        
        # Algorithm state
        self.ear_buffer = deque(maxlen=5)
        self.ear_threshold = None
        self.eye_closed_history = deque() # For PERCLOS windowing
        self.perclos_window_time = 30.0   # Shortened to 30s for demo

    def get_ear(self, landmarks, indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h_dist = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

    def run_demo(self):
        cap = cv2.VideoCapture(0)
        
        print(">>> [Phase 1] Calibration start! Please look at the camera for 5 seconds.")
        start_time = time.time()
        
        # --- Phase 1: Calibration ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            curr_time = time.time()
            elapsed = curr_time - start_time
            
            ear = 0.0
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left = self.get_ear(lm, self.LEFT_EYE, w, h)
                right = self.get_ear(lm, self.RIGHT_EYE, w, h)
                
                # EAR smoothing (short moving average)
                raw_ear = (left + right) / 2.0
                self.ear_buffer.append(raw_ear)
                ear = sum(self.ear_buffer) / len(self.ear_buffer)
                
                # Collect samples during calibration window
                if elapsed <= 5.0:
                    self.calib_data.append((elapsed, ear))

            # On-screen guidance
            if elapsed <= 5.0:
                msg = f"Calibrating... {5 - elapsed:.1f}s"
                color = (0, 255, 255) # yellow
            else:
                msg = "Calibration Done!"
                color = (0, 255, 0)   # green
                
            cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Logic Visualization", frame)
            cv2.waitKey(1)
            
            if elapsed > 5.0:
                break
        
        # Compute adaptive threshold from calibration
        ear_values = [v[1] for v in self.calib_data]
        avg_ear = sum(ear_values) / len(ear_values)
        self.ear_threshold = avg_ear * 0.8
        print(f">>> Calibration done. Avg: {avg_ear:.3f}, Threshold: {self.ear_threshold:.3f}")
        time.sleep(1)

        # --- Phase 2: PERCLOS demo ---
        print(f">>> [Phase 2] PERCLOS test for {int(self.perclos_window_time)}s. Blink or close your eyes.")
        perclos_start = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            curr_time = time.time()
            elapsed_total = curr_time - start_time      # 전체 시간
            elapsed_perclos = curr_time - perclos_start # 2단계 시간
            
            ear = 0.0
            perclos_score = 0.0
            is_closed = False
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left = self.get_ear(lm, self.LEFT_EYE, w, h)
                right = self.get_ear(lm, self.RIGHT_EYE, w, h)
                
                raw_ear = (left + right) / 2.0
                self.ear_buffer.append(raw_ear)
                ear = sum(self.ear_buffer) / len(self.ear_buffer)
                
                # Detect closure with adaptive EAR threshold
                is_closed = ear < self.ear_threshold
                
                # Update sliding window
                self.eye_closed_history.append((curr_time, is_closed))
                
                # Remove samples outside the window
                while self.eye_closed_history and (curr_time - self.eye_closed_history[0][0] > self.perclos_window_time):
                    self.eye_closed_history.popleft()
                
                # Compute PERCLOS (% closed within window)
                if self.eye_closed_history:
                    closed_frames = sum(1 for _, closed in self.eye_closed_history if closed)
                    total_frames = len(self.eye_closed_history)
                    perclos_score = (closed_frames / total_frames) * 100

                # Log for later visualization
                self.perclos_data["time"].append(elapsed_total)
                self.perclos_data["ear"].append(ear)
                self.perclos_data["is_closed"].append(1 if is_closed else 0)
                self.perclos_data["perclos_val"].append(perclos_score)

            # On-screen status
            status = "EYES CLOSED" if is_closed else "EYES OPEN"
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            
            cv2.putText(frame, f"Mode: PERCLOS Test ({int(self.perclos_window_time - elapsed_perclos)}s left)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.3f} | Thresh: {self.ear_threshold:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Status: {status}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"PERCLOS: {perclos_score:.1f}%", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Progress bar for remaining time
            bar_width = int((elapsed_perclos / self.perclos_window_time) * w)
            cv2.rectangle(frame, (0, h-10), (bar_width, h), (0, 255, 0), -1)

            cv2.imshow("Logic Visualization", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_perclos > self.perclos_window_time:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(">>> Data collection finished. Rendering graphs...")
        self.plot_graphs()

    def plot_graphs(self):

        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Figure 1: Calibration Result
        times_calib = [t for t, _ in self.calib_data]
        ears_calib = [e for _, e in self.calib_data]
        avg_ear = self.ear_threshold / 0.8
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(times_calib, ears_calib, label='Real-time EAR', color='royalblue', alpha=0.7)
        ax1.axhline(y=avg_ear, color='green', linestyle='--', linewidth=2, label=f'Average EAR ({avg_ear:.3f})')
        ax1.axhline(y=self.ear_threshold, color='red', linestyle='-', linewidth=2, label=f'Adaptive Threshold ({self.ear_threshold:.3f})')
        
        # Shade calibration interval (0–5s)
        ax1.axvspan(0, 5, color='yellow', alpha=0.1, label='Calibration Phase (5s)')
        
        ax1.set_title(f'Logic 1: Adaptive Threshold Calibration (User: You)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Eye Aspect Ratio (EAR)')
        ax1.legend(loc='lower left')
        ax1.set_ylim(0, max(ears_calib)*1.2)
        
        # Figure 2: PERCLOS Sliding Window Logic
        t_p = self.perclos_data["time"]
        ear_p = self.perclos_data["ear"]
        closed_p = self.perclos_data["is_closed"]
        perc_val = self.perclos_data["perclos_val"]
        
        fig2, (ax2_top, ax2_mid, ax2_bot) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        plt.subplots_adjust(hspace=0.3)
        
        # 2-1. EAR & Threshold
        ax2_top.plot(t_p, ear_p, color='gray', alpha=0.6, label='EAR Signal')
        ax2_top.axhline(y=self.ear_threshold, color='red', linestyle='--', label='Threshold')
        ax2_top.fill_between(t_p, 0, ear_p, where=[e < self.ear_threshold for e in ear_p], 
                             color='red', alpha=0.3, label='Detected Closed')
        ax2_top.set_ylabel('EAR')
        ax2_top.set_title('Step A: Detect Eye Closure based on Threshold', fontweight='bold')
        ax2_top.legend(loc='upper right')

        # 2-2. Sliding Window Binary Map
        binary_matrix = np.array([closed_p])
        ax2_mid.imshow(binary_matrix, aspect='auto', cmap='Reds', extent=[min(t_p), max(t_p), 0, 1], vmin=0, vmax=1)
        ax2_mid.set_yticks([])
        ax2_mid.set_ylabel('Window State')
        ax2_mid.set_title('Step B: Binary State in Sliding Window (Red=Closed)', fontweight='bold')
        
        # 2-3. PERCLOS % Curve
        ax2_bot.plot(t_p, perc_val, color='darkorange', linewidth=2, label='PERCLOS %')
        # Reference line for drowsiness (20%)
        ax2_bot.axhline(y=20, color='red', linestyle=':', label='Drowsy Limit (20%)')
        # Highlight above-threshold region
        ax2_bot.fill_between(t_p, 20, perc_val, where=[p >= 20 for p in perc_val], color='red', alpha=0.3)
        
        ax2_bot.set_ylabel('PERCLOS (%)')
        ax2_bot.set_xlabel('Time (seconds)')
        ax2_bot.set_title('Step C: Real-time PERCLOS Calculation', fontweight='bold')
        ax2_bot.set_ylim(0, 100)
        ax2_bot.legend()

        plt.show()

if __name__ == "__main__":
    visualizer = LogicVisualizer()
    visualizer.run_demo()
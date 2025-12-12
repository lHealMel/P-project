import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

class SmoothingVisualizer:
    def __init__(self):
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices (includes 8-point MAR set)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14, 81, 178, 311, 402]

        # 3D model points for PnP (arbitrary face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # nose tip
            (0.0, -330.0, -65.0),        # chin
            (-225.0, 170.0, -135.0),     # left eye corner
            (225.0, 170.0, -135.0),      # right eye corner
            (-150.0, -150.0, -125.0),    # left mouth corner
            (150.0, -150.0, -125.0)      # right mouth corner
        ])

        # Smoothing buffers (mirrors main implementation)
        self.ear_buffer = deque(maxlen=5)
        self.mar_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=10)
        
        self.pitch_ema = None
        self.pitch_alpha = 0.7
        
        self.mar_ema = None
        self.mar_alpha = 0.6
        
        # Time-series log for plotting raw vs smoothed signals
        self.data_log = {
            "time": [],
            "raw_ear": [], "smooth_ear": [],
            "raw_mar": [], "smooth_mar": [],
            "raw_pitch": [], "smooth_pitch": []
        }
        self.start_time = None

    # EAR calculation 
    def get_ear(self, landmarks, indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h_dist = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

    # MAR calculation (8-point set)
    def get_mar(self, landmarks, indices, w, h):
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])
        horiz = np.linalg.norm(pt(61) - pt(291))
        verticals = [
            np.linalg.norm(pt(13) - pt(14)),
            np.linalg.norm(pt(81) - pt(178)),
            np.linalg.norm(pt(311) - pt(402))
        ]
        return (sum(verticals) / len(verticals)) / horiz if horiz != 0 else 0

    # Head pose (PnP) calculation
    def get_head_pose(self, landmarks, w, h):
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[291].x * w, landmarks[291].y * h),
            (landmarks[61].x * w, landmarks[61].y * h)
        ], dtype="double")
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return None
        
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0] * 360 # Pitch

    def process_frame(self, image):
        if self.start_time is None: self.start_time = time.time()
        
        img_h, img_w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            current_time = time.time() - self.start_time
            
            # 1. EAR (moving average only)
            left = self.get_ear(landmarks, self.LEFT_EYE, img_w, img_h)
            right = self.get_ear(landmarks, self.RIGHT_EYE, img_w, img_h)
            raw_ear = (left + right) / 2.0
            
            self.ear_buffer.append(raw_ear)
            smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            
            # 2. MAR (moving average + EMA)
            raw_mar = self.get_mar(landmarks, self.MOUTH, img_w, img_h)
            self.mar_buffer.append(raw_mar)
            ma_mar = sum(self.mar_buffer) / len(self.mar_buffer)
            
            if self.mar_ema is None: self.mar_ema = ma_mar
            else: self.mar_ema = self.mar_alpha * ma_mar + (1 - self.mar_alpha) * self.mar_ema
            smooth_mar = self.mar_ema
            
            # 3. Pitch (moving average + EMA)
            raw_pitch = self.get_head_pose(landmarks, img_w, img_h)
            if raw_pitch is None: raw_pitch = 0
            
            self.pitch_buffer.append(raw_pitch)
            ma_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer)
            
            if self.pitch_ema is None: self.pitch_ema = ma_pitch
            else: self.pitch_ema = self.pitch_alpha * ma_pitch + (1 - self.pitch_alpha) * self.pitch_ema
            smooth_pitch = self.pitch_ema
            
            # Persist time-series for plotting
            self.data_log["time"].append(current_time)
            self.data_log["raw_ear"].append(raw_ear)
            self.data_log["smooth_ear"].append(smooth_ear)
            self.data_log["raw_mar"].append(raw_mar)
            self.data_log["smooth_mar"].append(smooth_mar)
            self.data_log["raw_pitch"].append(raw_pitch)
            self.data_log["smooth_pitch"].append(smooth_pitch)
            
            return True
        return False

    def plot_results(self):
        if not self.data_log["time"]:
            print("No data collected; nothing to plot.")
            return

        t = self.data_log["time"]
        
        plt.figure(figsize=(12, 10))
        plt.suptitle("Signal Processing Effect: Raw vs Smoothed", fontsize=16)
        
        # EAR Plot
        plt.subplot(3, 1, 1)
        plt.plot(t, self.data_log["raw_ear"], label='Raw EAR', color='blue', alpha=0.3)
        plt.plot(t, self.data_log["smooth_ear"], label='Smoothed EAR (MA)', color='red', linewidth=2)
        plt.title('Eye Aspect Ratio (EAR)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        # MAR Plot
        plt.subplot(3, 1, 2)
        plt.plot(t, self.data_log["raw_mar"], label='Raw MAR', color='green', alpha=0.3)
        plt.plot(t, self.data_log["smooth_mar"], label='Smoothed MAR (MA+EMA)', color='darkgreen', linewidth=2)
        plt.title('Mouth Aspect Ratio (MAR)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        # Pitch Plot
        plt.subplot(3, 1, 3)
        plt.plot(t, self.data_log["raw_pitch"], label='Raw Pitch', color='purple', alpha=0.3)
        plt.plot(t, self.data_log["smooth_pitch"], label='Smoothed Pitch (MA+EMA)', color='magenta', linewidth=2)
        plt.title('Head Pitch (PnP)')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.show()

# --- Main ---
def main():
    visualizer = SmoothingVisualizer()
    cap = cv2.VideoCapture(0)
    
    print(">>> Collecting data... blink or move your head while looking at the camera.")
    print(">>> Press 'q' to stop and plot the results.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        visualizer.process_frame(frame)
        
        cv2.putText(frame, "Recording Data... Press 'q' to Plot", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Signal Processing Visualization", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    print(">>> Rendering plots...")
    visualizer.plot_results()

if __name__ == "__main__":
    main()
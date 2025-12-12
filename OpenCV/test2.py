import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class RealtimePlotter:
    def __init__(self, title, max_len=100, w=400, h=100, y_range=(0, 1)):
        self.title = title
        self.w = w
        self.h = h
        self.max_len = max_len
        self.y_min, self.y_max = y_range
        
        # Buffers for raw and smoothed series
        self.raw_data = deque(maxlen=max_len)
        self.smooth_data = deque(maxlen=max_len)
        
        # Pre-rendered black background for plotting
        self.bg = np.zeros((h, w, 3), dtype=np.uint8)

    def update(self, raw, smooth):
        self.raw_data.append(raw)
        self.smooth_data.append(smooth)

    def draw(self):
        # Reset background
        img = self.bg.copy()
        
        # Border and title
        cv2.rectangle(img, (0, 0), (self.w-1, self.h-1), (50, 50, 50), 1)
        cv2.putText(img, self.title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if len(self.raw_data) < 2:
            return img

        # Map data value to (x, y) pixel coordinate
        def to_xy(i, val):
            x = int(i * (self.w / self.max_len))
            # 정규화 (Value -> 0~1) -> 픽셀 좌표 (H -> 0)
            norm_val = (val - self.y_min) / (self.y_max - self.y_min + 1e-6)
            y = int(self.h - (norm_val * self.h))
            y = max(0, min(self.h - 1, y)) # 클리핑
            return (x, y)

        # Draw lines for raw and smoothed series
        for i in range(1, len(self.raw_data)):
            # Raw data (thin blue)
            p1 = to_xy(i-1, self.raw_data[i-1])
            p2 = to_xy(i, self.raw_data[i])
            cv2.line(img, p1, p2, (255, 100, 100), 1) # BGR: 파랑

            # Smoothed data (thicker red)
            p1_s = to_xy(i-1, self.smooth_data[i-1])
            p2_s = to_xy(i, self.smooth_data[i])
            cv2.line(img, p1_s, p2_s, (0, 0, 255), 2) # BGR: 빨강

        # Current values overlay
        curr_raw = self.raw_data[-1]
        curr_smooth = self.smooth_data[-1]
        cv2.putText(img, f"Raw: {curr_raw:.2f}", (self.w - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 150), 1)
        cv2.putText(img, f"Smth: {curr_smooth:.2f}", (self.w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        return img

class DriverMonitorWithGraph:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        
        # Landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14, 81, 178, 311, 402]
        
        # PnP model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

        # Smoothing variables
        self.ear_buffer = deque(maxlen=5)
        self.mar_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=10)
        self.pitch_ema = None
        self.pitch_alpha = 0.7
        self.mar_ema = None
        self.mar_alpha = 0.6

        # Real-time plotters (value ranges tuned for typical signals)
        # EAR: typically ~0.15–0.35
        self.plotter_ear = RealtimePlotter("EAR (Eye)", y_range=(0.1, 0.4))
        # MAR: typically 0.0–0.8 (grows on yawn)
        self.plotter_mar = RealtimePlotter("MAR (Mouth)", y_range=(0.0, 1.0))
        # Pitch: typically -20–+40 deg
        self.plotter_pitch = RealtimePlotter("Pitch (Head)", y_range=(-30, 50))

    # --- Measurement helpers ---
    def get_ear(self, landmarks, indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h_dist = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

    def get_mar(self, landmarks, indices, w, h):
        def pt(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        horiz = np.linalg.norm(pt(61) - pt(291))
        verticals = [np.linalg.norm(pt(13)-pt(14)), np.linalg.norm(pt(81)-pt(178)), np.linalg.norm(pt(311)-pt(402))]
        return (sum(verticals)/len(verticals))/horiz if horiz!=0 else 0

    def get_head_pose(self, landmarks, w, h):
        image_points = np.array([
            (landmarks[1].x*w, landmarks[1].y*h), (landmarks[152].x*w, landmarks[152].y*h),
            (landmarks[263].x*w, landmarks[263].y*h), (landmarks[33].x*w, landmarks[33].y*h),
            (landmarks[291].x*w, landmarks[291].y*h), (landmarks[61].x*w, landmarks[61].y*h)
        ], dtype="double")
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[0]

    def process(self, frame):
        h, w, _ = frame.shape
        # Match graph widths to the video width
        self.plotter_ear.w = w // 3
        self.plotter_mar.w = w // 3
        self.plotter_pitch.w = w // 3 # fill remaining space proportionally
        
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        raw_ear, smooth_ear = 0, 0
        raw_mar, smooth_mar = 0, 0
        raw_pitch, smooth_pitch = 0, 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. EAR
            l_ear = self.get_ear(landmarks, self.LEFT_EYE, w, h)
            r_ear = self.get_ear(landmarks, self.RIGHT_EYE, w, h)
            raw_ear = (l_ear + r_ear) / 2.0
            self.ear_buffer.append(raw_ear)
            smooth_ear = sum(self.ear_buffer) / len(self.ear_buffer)

            # 2. MAR
            raw_mar = self.get_mar(landmarks, self.MOUTH, w, h)
            self.mar_buffer.append(raw_mar)
            ma_mar = sum(self.mar_buffer) / len(self.mar_buffer)
            if self.mar_ema is None: self.mar_ema = ma_mar
            else: self.mar_ema = self.mar_alpha * ma_mar + (1 - self.mar_alpha) * self.mar_ema
            smooth_mar = self.mar_ema

            # 3. Pitch
            raw_pitch = self.get_head_pose(landmarks, w, h)
            self.pitch_buffer.append(raw_pitch)
            ma_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer)
            if self.pitch_ema is None: self.pitch_ema = ma_pitch
            else: self.pitch_ema = self.pitch_alpha * ma_pitch + (1 - self.pitch_alpha) * self.pitch_ema
            smooth_pitch = self.pitch_ema

        # Push data to plot buffers
        self.plotter_ear.update(raw_ear, smooth_ear)
        self.plotter_mar.update(raw_mar, smooth_mar)
        self.plotter_pitch.update(raw_pitch, smooth_pitch)

        # Render graph images
        g1 = self.plotter_ear.draw()
        g2 = self.plotter_mar.draw()
        g3 = self.plotter_pitch.draw()
        
        # Concatenate three graphs horizontally (resize to align widths)
        target_w = w // 3
        g1 = cv2.resize(g1, (target_w, 150))
        g2 = cv2.resize(g2, (target_w, 150))
        remaining_w = w - (target_w * 2)
        g3 = cv2.resize(g3, (remaining_w, 150))
        
        graphs = np.hstack([g1, g2, g3])
        
        # Stack original frame (top) and graphs (bottom)
        combined = np.vstack([frame, graphs])
        
        return combined

# --- Main ---
def main():
    monitor = DriverMonitorWithGraph()
    cap = cv2.VideoCapture(0)
    
    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # set window name and make it adjustable
    window_name = "Real-time Signal Smoothing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

    print(">>> Running... (Blue: Raw / Red: Smoothed)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Run inference and merge graphs
        output_frame = monitor.process(frame)
        
        # Optional upscaling for display
        h, w = output_frame.shape[:2]
        scale_factor = 1.5  #
        new_dim = (int(w * scale_factor), int(h * scale_factor))
        resized_output = cv2.resize(output_frame, new_dim, interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow(window_name, resized_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class DriverMonitorCV:
    def __init__(self):

        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh

        # Key parameters
        # - max_num_faces: number of faces to track
        # - refine_landmarks: enables refined eye/lip landmarks
        # - min_detection_confidence: minimum detection confidence
        # - min_tracking_confidence: minimum tracking confidence
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark index sets (MediaPipe FaceMesh)
        # Example: LEFT_EYE starts with 33, which is the outer left eye corner.
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # MOUTH: 8-point set for MAR (mouth aspect ratio)
        # 61, 291: left/right mouth corners
        # 13, 14: upper/lower inner center
        # 81, 178: upper/lower inner (left)
        # 311, 402: upper/lower inner (right)
        self.MOUTH = [61, 291, 13, 14, 81, 178, 311, 402]

        # 3D model points for PnP (arbitrary scale) in face coordinate space
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 코 끝
            (0.0, -330.0, -65.0),        # 턱
            (-225.0, 170.0, -135.0),     # 왼쪽 눈 바깥꼬리
            (225.0, 170.0, -135.0),      # 오른쪽 눈 바깥꼬리
            (-150.0, -150.0, -125.0),    # 왼쪽 입꼬리
            (150.0, -150.0, -125.0)      # 오른쪽 입꼬리
        ])

        # State flags and timers
        self.is_calibrated = False
        self.calibration_start_time = None
        self.calibration_duration = 5.0 # Seconds to collect baseline EAR
        self.ear_list = []              # Stores EAR samples during calibration
        
        # Thresholds (updated after calibration)
        self.EAR_THRESHOLD = 0.25  # Initial EAR cutoff (personalized after calibration)
        self.MAR_THRESHOLD = 0.6   # Initial MAR cutoff
        self.PITCH_THRESHOLD = 25  # Pitch deg: sustained above this → head drop
        self.HEAD_DROP_FRAMES = 12 # Frames pitch must stay high to flag head drop
        self.head_drop_counter = 0
        
        # Buffers for PERCLOS and signal smoothing
        self.eye_closed_history = deque()       # (timestamp, closed?) for PERCLOS window
        self.ear_buffer = deque(maxlen=5)       # Short MA for EAR
        self.mar_buffer = deque(maxlen=5)       # Short MA for MAR
        self.pitch_buffer = deque(maxlen=10)    # Short MA for pitch
        self.perclos_window = 60                # PERCLOS window in seconds
        self.pitch_ema = None                   # EMA for pitch
        self.pitch_alpha = 0.7                  # EMA weight (higher → more recent)
        self.mar_ema = None                     # EMA for MAR
        self.mar_alpha = 0.6                    # EMA weight for MAR smoothing

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_ear(self, landmarks, indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h_dist = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

    def get_mar(self, landmarks, indices, w, h):
        """
        8-point MAR:
        - Horizontal: mouth corners (61–291)
        - Vertical: mean of three upper/lower inner pairs (13–14, 81–178, 311–402)
        """
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        horiz = np.linalg.norm(pt(61) - pt(291))
        verticals = [
            np.linalg.norm(pt(13) - pt(14)),
            np.linalg.norm(pt(81) - pt(178)),
            np.linalg.norm(pt(311) - pt(402))
        ]
        vert = sum(verticals) / len(verticals)

        return vert / horiz if horiz != 0 else 0

    def get_head_pose(self, landmarks, w, h):
        """
        Estimate head rotation (Euler: Pitch, Yaw, Roll) using SolvePnP.
        """
        # 2D image points (nose, chin, eye corners, mouth corners) mapped to 3D model points
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),      # 코 끝
            (landmarks[152].x * w, landmarks[152].y * h),  # 턱
            (landmarks[263].x * w, landmarks[263].y * h),  # 왼쪽 눈 바깥꼬리
            (landmarks[33].x * w, landmarks[33].y * h),    # 오른쪽 눈 바깥꼬리
            (landmarks[291].x * w, landmarks[291].y * h),  # 왼쪽 입꼬리
            (landmarks[61].x * w, landmarks[61].y * h)     # 오른쪽 입꼬리
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1)) # Assume no lens distortion

        # Solve PnP to obtain rotation/translation
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        # Rotation vector → rotation matrix → Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # angles: [Pitch(X), Yaw(Y), Roll(Z)]
        # Pitch: down(+), up(-)
        return angles[0] * 360, angles[1] * 360, angles[2] * 360 # scale to degrees

    def process_frame(self, image):
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        data = {
            "ear": 0.0, "mar": 0.0, "perclos": 0.0, 
            "pitch": 0.0, "status": "No Face"
        }

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. EAR with short moving average to suppress jitter
            left_ear = self.get_ear(landmarks, self.LEFT_EYE, img_w, img_h)
            right_ear = self.get_ear(landmarks, self.RIGHT_EYE, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            self.ear_buffer.append(avg_ear)
            smoothed_ear = sum(self.ear_buffer) / len(self.ear_buffer)
            data["ear"] = smoothed_ear

            # 2. MAR (mouth opening) with MA + EMA smoothing
            raw_mar = self.get_mar(landmarks, self.MOUTH, img_w, img_h)
            self.mar_buffer.append(raw_mar)
            smoothed_mar = sum(self.mar_buffer) / len(self.mar_buffer)

            if self.mar_ema is None:
                self.mar_ema = smoothed_mar
            else:
                self.mar_ema = self.mar_alpha * smoothed_mar + (1 - self.mar_alpha) * self.mar_ema

            data["mar"] = self.mar_ema

            # 3. Head pose via SolvePnP (pitch/yaw/roll)
            pitch, yaw, roll = self.get_head_pose(landmarks, img_w, img_h)
            if pitch is not None:
                self.pitch_buffer.append(pitch)
                smoothed_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer)

                # Extra EMA smoothing to reduce head-drop false positives
                if self.pitch_ema is None:
                    self.pitch_ema = smoothed_pitch
                else:
                    self.pitch_ema = self.pitch_alpha * smoothed_pitch + (1 - self.pitch_alpha) * self.pitch_ema

                data["pitch"] = self.pitch_ema

            # --- Branch: calibration vs monitoring ---
            current_time = time.time()
            
            if not self.is_calibrated:
                # [Calibration] collect baseline EAR for personalization
                if self.calibration_start_time is None:
                    self.calibration_start_time = current_time
                
                elapsed = current_time - self.calibration_start_time
                self.ear_list.append(smoothed_ear)
                
                data["status"] = f"Calibrating... {5 - int(elapsed)}s"
                
                if elapsed > self.calibration_duration:
                    # Personalize EAR threshold to 80% of calibration mean
                    avg_calib_ear = sum(self.ear_list) / len(self.ear_list)
                    self.EAR_THRESHOLD = avg_calib_ear * 0.8 
                    self.is_calibrated = True
                    print(f"[INFO] Calibration Complete. New EAR Threshold: {self.EAR_THRESHOLD:.3f}")
            else:
                # [Monitoring] update PERCLOS and state
                is_closed = smoothed_ear < self.EAR_THRESHOLD
                self.eye_closed_history.append((current_time, is_closed))
                
                # Drop stale entries outside the PERCLOS window
                while self.eye_closed_history and (current_time - self.eye_closed_history[0][0] > self.perclos_window):
                    self.eye_closed_history.popleft()
                
                if self.eye_closed_history:
                    closed_count = sum(1 for _, closed in self.eye_closed_history if closed)
                    data["perclos"] = (closed_count / len(self.eye_closed_history)) * 100

                # Status aggregation
                warnings = []
                if is_closed: warnings.append("EYES CLOSED")
                if data["mar"] > self.MAR_THRESHOLD: warnings.append("YAWNING")

                # Head drop: rely on smoothed pitch + sustained frames to avoid spikes
                if self.pitch_ema > self.PITCH_THRESHOLD:
                    self.head_drop_counter += 1
                else:
                    self.head_drop_counter = 0
                if self.head_drop_counter >= self.HEAD_DROP_FRAMES:
                    warnings.append("HEAD DROP")

                data["status"] = ", ".join(warnings) if warnings else "NORMAL"

        return data

# --- Main ---
def main():
    detector = DriverMonitorCV()
    cap = cv2.VideoCapture(0)
    
    print(">>> Press 'q' to exit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Run inference and compute metrics for this frame
        data = detector.process_frame(frame)
        
        # On-screen debug overlay
        color = (0, 255, 0) if data["status"] == "NORMAL" or "Calibrating" in data["status"] else (0, 0, 255)
        
        # Text overlay logging
        cv2.putText(frame, f"State: {data['status']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"EAR: {data['ear']:.2f} (Thresh: {detector.EAR_THRESHOLD:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"PERCLOS: {data['perclos']:.1f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Head Pitch: {data['pitch']:.1f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Advanced Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
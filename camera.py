import cv2
import mediapipe as mp
import threading
import numpy as np
import time
from filters import KalmanFilter2D

class CameraTracker:
    def __init__(self, width=1280, height=720, webcam_index=0):
        self.width = width
        self.height = height
        self.webcam_index = webcam_index
        
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        
        # Output frame and hand data
        self.latest_frame = None
        self.hands_data = {}  # Format: {'Left': {'landmarks': np.array, 'confidence': float}, ...}
        self.camera_active = False
        
        # Initialize Kalman filters for 21 landmarks per hand
        # Using a dt of 1/60s because physics updates at 60Hz
        self.filters = {
            'Left': [KalmanFilter2D(dt=1.0/60.0, process_noise=0.012, measurement_noise=0.22) for _ in range(21)],
            'Right': [KalmanFilter2D(dt=1.0/60.0, process_noise=0.012, measurement_noise=0.22) for _ in range(21)]
        }
        
        # Tracking persistence
        self.loss_frames = {'Left': 100, 'Right': 100}
        self.max_loss_frames = 10  # Predict coordinates for up to 10 frames before dropping hand
        
        # Exponential moving average for Z coordinate smoothing (depth)
        self.z_ema = {
            'Left': np.zeros(21, dtype=np.float32),
            'Right': np.zeros(21, dtype=np.float32)
        }
        self.z_ema_alpha = 0.25

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def _run(self):
        # Attempt to initialize webcam with default backend
        print(f"[HEXIS Camera] Attempting to open camera {self.webcam_index} with default backend...")
        self.cap = cv2.VideoCapture(self.webcam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test if we can read a frame (using a small warm-up loop)
        has_frames = False
        if self.cap.isOpened():
            for _ in range(10):  # Up to 500ms
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    has_frames = True
                    break
                time.sleep(0.05)
                
            if not has_frames:
                print("[HEXIS Camera] Default backend failed to read frames. Releasing and trying DirectShow fallback...")
                self.cap.release()
                self.cap = None
                
        # If default backend failed, try DirectShow fallback
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        # Final validation with a warm-up loop to let the camera sensor initialize
        if not self.cap.isOpened():
            print(f"[HEXIS Camera] Warning: Could not open camera at index {self.webcam_index}.")
            self.camera_active = False
            self.latest_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            has_frames = False
            for _ in range(15):  # Up to 750ms for DirectShow/Hardware stabilization
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    has_frames = True
                    break
                time.sleep(0.05)
                
            if not has_frames:
                print(f"[HEXIS Camera] Warning: Camera is opened but returned empty frames at index {self.webcam_index}.")
                self.camera_active = False
                self.latest_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else:
                self.camera_active = True
                print(f"[HEXIS Camera] Camera started successfully at {self.width}x{self.height} (Index {self.webcam_index}).")

        # Initialize MediaPipe Hands
        import sys
        try:
            import mediapipe.python.solutions.hands as mp_hands
        except ModuleNotFoundError:
            try:
                import mediapipe.solutions.hands as mp_hands
            except ModuleNotFoundError:
                print("\n" + "="*80)
                print("[HEXIS CRITICAL] MediaPipe Solutions is missing in this Python environment.")
                print(f"Current Executable: {sys.executable}")
                print(f"Current Version   : {sys.version}")
                print("\n>>> TO RUN HEXIS SUCCESSFULLY, PLEASE LAUNCH WITH PYTHON 3.11:")
                print("    py -3.11 main.py")
                print("="*80 + "\n")
                self.camera_active = False
                # Fill latest frame with a warning banner image or black
                self.latest_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )

        last_time = time.time()
        
        while self.running:
            if self.camera_active:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Mirror horizontally for natural mirror behavior
                frame = cv2.flip(frame, 1)
                
                # Resize if camera did not respect requested resolution
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Time delta calculation for Kalman
                now = time.time()
                dt = now - last_time
                last_time = now
                
                # Update dt in Kalman filters
                for side in ['Left', 'Right']:
                    for f in self.filters[side]:
                        f.dt = dt

                detected_sides = set()
                new_hands_data = {}

                if results.multi_hand_landmarks and results.multi_handedness:
                    for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # MediaPipe labels are from hand anatomical perspective
                        # But since we flipped the frame horizontally, we should swap Left and Right
                        # to match the screen visual side (Left of screen vs Right of screen)
                        raw_label = handedness.classification[0].label  # 'Left' or 'Right'
                        label = 'Right' if raw_label == 'Left' else 'Left'
                        confidence = handedness.classification[0].score
                        
                        detected_sides.add(label)
                        self.loss_frames[label] = 0
                        
                        # Process landmarks
                        coords_2d = []
                        coords_3d = [] # We keep Z coordinate
                        
                        for idx, lm in enumerate(landmarks.landmark):
                            # Pixel coordinates
                            px = lm.x * self.width
                            py = lm.y * self.height
                            pz = lm.z * self.width  # Scale Z depth isotropically with width
                            
                            # Apply Kalman filter to X, Y
                            kf = self.filters[label][idx]
                            smoothed_xy = kf.update(np.array([px, py], dtype=np.float32))
                            
                            # Smooth Z with EMA
                            if self.loss_frames[label] == 0 and not kf.initialized:
                                self.z_ema[label][idx] = pz
                            else:
                                self.z_ema[label][idx] = (self.z_ema_alpha * pz + 
                                                          (1 - self.z_ema_alpha) * self.z_ema[label][idx])
                                
                            coords_2d.append(smoothed_xy)
                            coords_3d.append([smoothed_xy[0], smoothed_xy[1], self.z_ema[label][idx]])
                        
                        new_hands_data[label] = {
                            'landmarks': np.array(coords_3d, dtype=np.float32),
                            'confidence': confidence,
                            'active': True
                        }

                # Handle tracking loss: predict position for active hands that disappeared briefly
                for side in ['Left', 'Right']:
                    if side not in detected_sides:
                        self.loss_frames[side] += 1
                        if self.loss_frames[side] <= self.max_loss_frames:
                            # Use Kalman prediction to keep tracing trajectory
                            coords_3d = []
                            for idx, kf in enumerate(self.filters[side]):
                                predicted_xy = kf.predict()
                                coords_3d.append([predicted_xy[0], predicted_xy[1], self.z_ema[side][idx]])
                            
                            # Interpolate confidence decay
                            conf = 0.5 * (1.0 - (self.loss_frames[side] / self.max_loss_frames))
                            new_hands_data[side] = {
                                'landmarks': np.array(coords_3d, dtype=np.float32),
                                'confidence': max(0.01, conf),
                                'active': True
                            }
                        else:
                            # Tracking completely lost, reset filters
                            for kf in self.filters[side]:
                                kf.reset()
                            new_hands_data[side] = {
                                'landmarks': np.zeros((21, 3), dtype=np.float32),
                                'confidence': 0.0,
                                'active': False
                            }
                
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.hands_data = new_hands_data
                    
                # Small sleep to throttle webcam read (caps CPU usage)
                time.sleep(0.005)
            else:
                # Camera is inactive, sleep to save CPU resources
                time.sleep(0.03)

    def get_data(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            hands = self.hands_data.copy()
            return frame, hands

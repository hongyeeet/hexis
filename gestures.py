import numpy as np
import math

def calculate_angle(v1, v2):
    """
    Calculates the angle (in radians) between two 3D vectors.
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.arccos(cos_angle)

def calculate_finger_curls(landmarks):
    """
    Calculates curl depth (0.0 to 1.0) for each of the 5 fingers.
    landmarks: (21, 3) array of [x, y, z] hand coordinates
    Returns: np.array of 5 floats [thumb, index, middle, ring, pinky]
    """
    curls = np.zeros(5, dtype=np.float32)
    
    # 1. Thumb (CMC=1, MCP=2, IP=3, TIP=4)
    v1 = landmarks[2] - landmarks[1]
    v2 = landmarks[3] - landmarks[2]
    v3 = landmarks[4] - landmarks[3]
    a1 = calculate_angle(v1, v2)
    a2 = calculate_angle(v2, v3)
    # Thumb curls up to about 1.4 radians sum
    curls[0] = np.clip((a1 + a2) / 1.4, 0.0, 1.0)
    
    # 2. Index (MCP=5, PIP=6, DIP=7, TIP=8)
    v1 = landmarks[6] - landmarks[5]
    v2 = landmarks[7] - landmarks[6]
    v3 = landmarks[8] - landmarks[7]
    a1 = calculate_angle(v1, v2)
    a2 = calculate_angle(v2, v3)
    curls[1] = np.clip((a1 + a2) / 2.5, 0.0, 1.0)
    
    # 3. Middle (MCP=9, PIP=10, DIP=11, TIP=12)
    v1 = landmarks[10] - landmarks[9]
    v2 = landmarks[11] - landmarks[10]
    v3 = landmarks[12] - landmarks[11]
    a1 = calculate_angle(v1, v2)
    a2 = calculate_angle(v2, v3)
    curls[2] = np.clip((a1 + a2) / 2.5, 0.0, 1.0)
    
    # 4. Ring (MCP=13, PIP=14, DIP=15, TIP=16)
    v1 = landmarks[14] - landmarks[13]
    v2 = landmarks[15] - landmarks[14]
    v3 = landmarks[16] - landmarks[15]
    a1 = calculate_angle(v1, v2)
    a2 = calculate_angle(v2, v3)
    curls[3] = np.clip((a1 + a2) / 2.5, 0.0, 1.0)
    
    # 5. Pinky (MCP=17, PIP=18, DIP=19, TIP=20)
    v1 = landmarks[18] - landmarks[17]
    v2 = landmarks[19] - landmarks[18]
    v3 = landmarks[20] - landmarks[19]
    a1 = calculate_angle(v1, v2)
    a2 = calculate_angle(v2, v3)
    curls[4] = np.clip((a1 + a2) / 2.5, 0.0, 1.0)
    
    return curls

def calculate_hand_scale(landmarks):
    """
    Computes a scale factor based on hand size (wrist to middle finger MCP)
    to normalize spatial calculations.
    """
    return max(1.0, float(np.linalg.norm(landmarks[9] - landmarks[0])))

def calculate_pinch_states(landmarks):
    """
    Returns normalized pinch distances and pinch detection states for fingers index, middle, ring, pinky.
    A pinch is defined when the fingertip is close to the thumb tip.
    """
    hand_scale = calculate_hand_scale(landmarks)
    thumb_tip = landmarks[4]
    
    tips = {
        'index': 8,
        'middle': 12,
        'ring': 16,
        'pinky': 20
    }
    
    pinch_dists = {}
    pinch_active = {}
    
    # Threshold for pinch: typical thumb-tip to finger-tip normalized distance < 0.28
    PINCH_THRESHOLD = 0.28
    
    for name, idx in tips.items():
        dist = np.linalg.norm(landmarks[idx] - thumb_tip) / hand_scale
        pinch_dists[name] = float(dist)
        pinch_active[name] = dist < PINCH_THRESHOLD
        
    return pinch_dists, pinch_active

def calculate_wrist_rotation(landmarks, side):
    """
    Returns the wrist roll/rotation angle in radians.
    Measures the tilt of the hand relative to the screen.
    For right hand, moving from palm down to palm up.
    """
    # Vector from Index MCP (5) to Pinky MCP (17)
    v_width = landmarks[17] - landmarks[5]
    
    # Roll angle in X-Y plane
    angle = math.atan2(v_width[1], v_width[0])
    
    # Adjust for side so left and right rotations align naturally
    if side == 'Left':
        # Reflect angle
        angle = math.pi - angle
        if angle > math.pi:
            angle -= 2 * math.pi
            
    return angle

class WristFlickDetector:
    def __init__(self, threshold=1200.0, cooldown=0.3):
        self.prev_pos = None
        self.threshold = threshold  # Pixels per second
        self.cooldown = cooldown
        self.cooldown_timer = 0.0
        
    def update(self, wrist_pos, dt):
        """
        wrist_pos: [x, y] coordinates of landmark 0
        """
        if self.cooldown_timer > 0.0:
            self.cooldown_timer -= dt
            
        if self.prev_pos is None or dt <= 0:
            self.prev_pos = wrist_pos.copy()
            return False, np.zeros(2, dtype=np.float32)
            
        # Calculate velocity
        vel = (wrist_pos[:2] - self.prev_pos[:2]) / dt
        speed = np.linalg.norm(vel)
        self.prev_pos = wrist_pos.copy()
        
        if speed > self.threshold and self.cooldown_timer <= 0.0:
            self.cooldown_timer = self.cooldown
            direction = vel / (speed + 1e-5)
            return True, direction
            
        return False, np.zeros(2, dtype=np.float32)

class GestureAnalyzer:
    def __init__(self):
        self.flick_detectors = {
            'Left': WristFlickDetector(),
            'Right': WristFlickDetector()
        }
        self.swipe_cooldown = 0.0
        self.prev_hand_dist = None
        self.clap_cooldown = 0.0
        
        # Track previous values for delta calculations
        self.prev_wrists = {'Left': None, 'Right': None}
        self.prev_rotations = {'Left': 0.0, 'Right': 0.0}
        self.hand_velocities = {
            'Left': np.zeros(2, dtype=np.float32),
            'Right': np.zeros(2, dtype=np.float32)
        }
        self.rotation_speeds = {'Left': 0.0, 'Right': 0.0}

    def process(self, hands_data, dt):
        """
        Processes both hands and updates gesture states.
        hands_data: dict of detected hands from camera tracker
        Returns: a dictionary of computed gesture metrics
        """
        gestures = {
            'Left': self._empty_hand_gestures(),
            'Right': self._empty_hand_gestures(),
            'swipe_swap': 0,
            'clap_singularity': False,
            'clap_pos': np.zeros(2, dtype=np.float32),
            'parallel_sweep': False,
            'parallel_sweep_dir': np.zeros(2, dtype=np.float32),
            'bowl_shape': False,
            'bowl_pos': np.zeros(2, dtype=np.float32),
            'wrists_close_spread': False,
            'gojo_infinity': False,
            'torch_raise': False,
            'torch_pos': np.zeros(2, dtype=np.float32),
        }
        
        # 1. Update individual hands
        active_sides = []
        for side in ['Left', 'Right']:
            hand = hands_data.get(side)
            if hand and hand.get('active') and hand['confidence'] > 0.1:
                landmarks = hand['landmarks']
                active_sides.append(side)
                
                curls = calculate_finger_curls(landmarks)
                pinch_dists, pinch_active = calculate_pinch_states(landmarks)
                rotation = calculate_wrist_rotation(landmarks, side)
                
                # Continuous velocity tracking
                wrist_pos = landmarks[0][:2]
                if self.prev_wrists[side] is not None and dt > 0:
                    vel = (wrist_pos - self.prev_wrists[side]) / dt
                    self.hand_velocities[side] = vel
                self.prev_wrists[side] = wrist_pos.copy()
                
                # Rotation speed tracking
                if dt > 0:
                    rot_diff = rotation - self.prev_rotations[side]
                    # Handle angle wrap-around (-pi to pi)
                    if rot_diff > math.pi:
                        rot_diff -= 2 * math.pi
                    elif rot_diff < -math.pi:
                        rot_diff += 2 * math.pi
                    self.rotation_speeds[side] = rot_diff / dt
                self.prev_rotations[side] = rotation
                
                # Flick detection
                is_flick, flick_dir = self.flick_detectors[side].update(landmarks[0], dt)
                
                gestures[side] = {
                    'active': True,
                    'landmarks': landmarks,
                    'curls': curls,
                    'pinch_dists': pinch_dists,
                    'pinch_active': pinch_active,
                    'rotation': rotation,
                    'rotation_speed': self.rotation_speeds[side],
                    'velocity': self.hand_velocities[side].copy(),
                    'flick': is_flick,
                    'flick_dir': flick_dir,
                    'palm_center': landmarks[9][:2],  # Middle finger MCP as palm center
                    'wrist': landmarks[0][:2]
                }
            else:
                self.prev_wrists[side] = None
                self.hand_velocities[side].fill(0.0)
                self.rotation_speeds[side] = 0.0

        # 2. Dual-hand gestures
        if len(active_sides) == 2:
            left_hand = gestures['Left']
            right_hand = gestures['Right']
            
            p1 = left_hand['palm_center']
            p2 = right_hand['palm_center']
            dist = np.linalg.norm(p1 - p2)
            
            # Singularity Clap: hands moving together quickly
            if self.clap_cooldown > 0.0:
                self.clap_cooldown -= dt
                
            if self.prev_hand_dist is not None and self.clap_cooldown <= 0.0:
                rel_speed = (dist - self.prev_hand_dist) / dt
                # Loosened: distance < 180px (was 120) and closing speed > 700 (was 1000)
                if dist < 180.0 and rel_speed < -700.0:
                    gestures['clap_singularity'] = True
                    gestures['clap_pos'] = (p1 + p2) * 0.5
                    self.clap_cooldown = 1.5
                    
            self.prev_hand_dist = dist
            
            # A. Parallel Sweep (Current Super)
            left_vel = left_hand['velocity']
            right_vel = right_hand['velocity']
            left_speed = np.linalg.norm(left_vel)
            right_speed = np.linalg.norm(right_vel)
            
            if left_speed > 350.0 and right_speed > 350.0:
                dot_prod = np.dot(left_vel, right_vel)
                cos_sim = dot_prod / (left_speed * right_speed + 1e-5)
                if cos_sim > 0.82:  # Parallel directions
                    gestures['parallel_sweep'] = True
                    gestures['parallel_sweep_dir'] = (left_vel / left_speed + right_vel / right_speed) * 0.5
            
            # B. Bowl Shape (Inferno Super)
            left_curl = np.mean(left_hand['curls'][1:])
            right_curl = np.mean(right_hand['curls'][1:])
            if (left_hand['wrist'][1] < 260.0 and right_hand['wrist'][1] < 260.0 and
                left_curl < 0.45 and right_curl < 0.45 and
                left_hand['palm_center'][1] < left_hand['wrist'][1] - 15.0 and
                right_hand['palm_center'][1] < right_hand['wrist'][1] - 15.0 and
                dist > 220.0):
                
                gestures['bowl_shape'] = True
                gestures['bowl_pos'] = (p1 + p2) * 0.5
                
            # C. Wrists Close (Arc Super)
            left_scale = calculate_hand_scale(left_hand['landmarks'])
            right_scale = calculate_hand_scale(right_hand['landmarks'])
            
            left_spread = np.linalg.norm(left_hand['landmarks'][8][:2] - left_hand['landmarks'][12][:2]) / left_scale
            right_spread = np.linalg.norm(right_hand['landmarks'][8][:2] - right_hand['landmarks'][12][:2]) / right_scale
            
            w_dist = np.linalg.norm(left_hand['wrist'] - right_hand['wrist'])
            if w_dist < 220.0 and left_spread > 0.60 and right_spread > 0.60:
                gestures['wrists_close_spread'] = True
            
            # D. Gojo Infinity Sign — crossed middle + ring fingers on BOTH hands
            # Gojo Satoru's "Infinity" activation: middle and ring fingertips crossed/touching,
            # while other fingers are loosely open or extended.
            def _hand_has_cross(hand_ref):
                lms = hand_ref['landmarks']
                hs  = calculate_hand_scale(lms)
                tip_mid  = lms[12][:2]   # middle tip
                tip_ring = lms[16][:2]   # ring tip
                tip_idx  = lms[8][:2]    # index tip  (should NOT also be touching ring)
                cross_dist = np.linalg.norm(tip_mid - tip_ring) / hs
                idx_dist   = np.linalg.norm(tip_idx  - tip_ring) / hs
                avg_c = float(np.mean(hand_ref['curls'][1:]))
                # Tips of middle+ring very close, not a full fist, index not also collapsed
                return cross_dist < 0.22 and idx_dist > cross_dist * 1.3 and avg_c < 0.80

            if _hand_has_cross(left_hand) and _hand_has_cross(right_hand):
                gestures['gojo_infinity'] = True

            # E. Torch Raise — Inferno Super replacement for bowl gesture
            # Both index + middle fingers extended upward, ring + pinky curled
            # Like holding two torches / devil-horn salute but pointing up
            l_c = left_hand['curls']
            r_c = right_hand['curls']
            l_torch = (l_c[1] < 0.32 and l_c[2] < 0.38 and l_c[3] > 0.52 and l_c[4] > 0.52)
            r_torch = (r_c[1] < 0.32 and r_c[2] < 0.38 and r_c[3] > 0.52 and r_c[4] > 0.52)
            if l_torch and r_torch and dist > 120.0:
                gestures['torch_raise'] = True
                gestures['torch_pos']   = (p1 + p2) * 0.5
                
        else:
            self.prev_hand_dist = None
            if self.clap_cooldown > 0.0:
                self.clap_cooldown -= dt

        # 3. Swipe-to-Swap mode cycle detection (X velocity > 1500 px/s)
        if self.swipe_cooldown > 0.0:
            self.swipe_cooldown -= dt
            
        if self.swipe_cooldown <= 0.0:
            for side in ['Left', 'Right']:
                if gestures[side]['active']:
                    vel_x = gestures[side]['velocity'][0]
                    if vel_x > 1400.0:    # Swipe Right (Cycle Forward)
                        gestures['swipe_swap'] = 1
                        self.swipe_cooldown = 0.8
                        print(f"[HEXIS Gesture] Swipe RIGHT detected ({vel_x:.1f} px/s) on {side} hand.")
                        break
                    elif vel_x < -1400.0:  # Swipe Left (Cycle Backward)
                        gestures['swipe_swap'] = -1
                        self.swipe_cooldown = 0.8
                        print(f"[HEXIS Gesture] Swipe LEFT detected ({vel_x:.1f} px/s) on {side} hand.")
                        break

        return gestures

    def _empty_hand_gestures(self):
        return {
            'active': False,
            'landmarks': np.zeros((21, 3), dtype=np.float32),
            'curls': np.zeros(5, dtype=np.float32),
            'pinch_dists': {'index': 1.0, 'middle': 1.0, 'ring': 1.0, 'pinky': 1.0},
            'pinch_active': {'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'rotation': 0.0,
            'rotation_speed': 0.0,
            'velocity': np.zeros(2, dtype=np.float32),
            'flick': False,
            'flick_dir': np.zeros(2, dtype=np.float32),
            'palm_center': np.zeros(2, dtype=np.float32),
            'wrist': np.zeros(2, dtype=np.float32)
        }

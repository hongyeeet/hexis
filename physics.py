import numpy as np
import random
import math

class ParticleSystem:
    def __init__(self, width=960, height=540, max_particles=5000):
        self.width = width
        self.height = height
        self.max_particles = max_particles
        
        # NumPy-vectorized arrays for particles
        self.pos = np.zeros((max_particles, 2), dtype=np.float32)
        self.prev_pos = np.zeros((max_particles, 2), dtype=np.float32)
        self.vel = np.zeros((max_particles, 2), dtype=np.float32)
        self.color = np.zeros((max_particles, 3), dtype=np.uint8) # RGB
        self.size = np.zeros(max_particles, dtype=np.float32)
        self.alpha = np.zeros(max_particles, dtype=np.float32)
        self.life = np.zeros(max_particles, dtype=np.float32)      # 1.0 (spawned) -> 0.0 (dead)
        self.decay = np.zeros(max_particles, dtype=np.float32)
        self.active = np.zeros(max_particles, dtype=bool)
        
        # mode_attrib[i, 0]: type (0: normal, 1: launched/charged projectile, 2: smoke, 3: explosion/shockwave shard)
        # mode_attrib[i, 1]: orbit phase/angle or custom parameter
        # mode_attrib[i, 2]: original speed or custom parameter
        self.mode_attrib = np.zeros((max_particles, 3), dtype=np.float32)
        
        # State machine
        self.mode = 1  # 1: SURGE, 2: INFERNO, 3: VOID, 4: CURRENT, 5: ARC
        self.next_mode = None
        self.transition_timer = 0.0
        self.transition_duration = 0.5  # 0.5s dissolve
        
        # Super ability state machine (shared across modes)
        self.super_active = False
        self.super_type = 0  # 1: Surge, 2: Inferno, 3: Void, 4: Current, 5: Arc
        self.super_pos = np.zeros(2, dtype=np.float32)
        self.super_timer = 0.0
        self.super_stage = 0  # 0: charging/active, 1: detonating/cooling
        self.super_vel = np.zeros(2, dtype=np.float32)
        self.super_charge = 0.0
        
        # Legacy compatibility for main.py references
        self.singularity_active = False
        self.singularity_pos = np.zeros(2, dtype=np.float32)
        self.singularity_timer = 0.0
        self.singularity_stage = 0
        
        # Track pinch objects
        self.grabbed_particles = {'Left': None, 'Right': None}  # Stores mask of indices
        self.grab_scale = {'Left': 1.0, 'Right': 1.0}
        self.grab_rotation = {'Left': 0.0, 'Right': 0.0}
        self.prev_rotation = {'Left': 0.0, 'Right': 0.0}
        self.shake_trigger = 0.0
        
        # Charge accumulator per hand (tracks how long gesture is held, 0.0–3.0 s)
        self.charge_time = {'Left': 0.0, 'Right': 0.0}
        
        # Elapsed time (seconds) for smooth noise fields
        self._time = 0.0
        
        # Pre-fill some particles for SURGE
        self.reset_mode_particles(self.mode)

    def reset_mode_particles(self, mode):
        self.active.fill(False)
        self.mode = mode

    def spawn_ambient(self, num):
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return
        count = min(num, len(inactive))
        idx = inactive[:count]
        
        self.pos[idx, 0] = np.random.uniform(0, self.width, count)
        self.pos[idx, 1] = np.random.uniform(0, self.height, count)
        self.vel[idx, 0] = np.random.uniform(-40, 40, count)
        self.vel[idx, 1] = np.random.uniform(-40, 40, count)
        
        if self.mode == 1:  # SURGE
            self.color[idx] = [0, 212, 255]  # #00d4ff
            self.size[idx] = np.random.uniform(1.5, 3.5, count)
        elif self.mode == 3:  # VOID
            # Deep purple/violet edges
            self.color[idx] = [138, 43, 226]  # Blue-violet
            self.size[idx] = np.random.uniform(2.0, 4.0, count)
        elif self.mode == 4:  # CURRENT
            # Blue-green bioluminescent
            self.color[idx] = [0, 250, 154]  # Medium spring green
            self.size[idx] = np.random.uniform(2.0, 5.0, count)
        elif self.mode == 5:  # ARC
            self.color[idx] = [200, 230, 255]  # Very cold blue-white
            self.size[idx] = np.random.uniform(1.0, 3.0, count)
            
        self.life[idx] = np.random.uniform(0.6, 1.0, count)
        self.decay[idx] = np.random.uniform(0.05, 0.15, count)
        self.alpha[idx] = 255.0
        self.mode_attrib[idx] = 0.0
        self.prev_pos[idx] = self.pos[idx].copy()
        self.active[idx] = True

    def trigger_mode_switch(self, next_mode):
        if self.transition_timer <= 0.0:
            self.next_mode = next_mode
            self.transition_timer = self.transition_duration
            self.shake_trigger = max(self.shake_trigger, 7.0)
            
    def _trigger_mode_switch_burst(self, pos, mode):
        count = min(120, np.sum(~self.active))
        if count == 0:
            return
        angles = np.random.uniform(0, 2*math.pi, count)
        speeds = np.random.uniform(150, 400, count)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        
        colors = np.zeros((count, 3), dtype=np.uint8)
        if mode == 1:    # Surge
            colors[:, 0] = 0
            colors[:, 1] = np.random.randint(180, 230, count)
            colors[:, 2] = 255
        elif mode == 2:  # Inferno
            colors[:, 0] = 255
            colors[:, 1] = np.random.randint(60, 150, count)
            colors[:, 2] = 0
        elif mode == 3:  # Void
            colors[:, 0] = np.random.randint(100, 160, count)
            colors[:, 1] = 0
            colors[:, 2] = 255
        elif mode == 4:  # Current
            colors[:, 0] = 0
            colors[:, 1] = 255
            colors[:, 2] = np.random.randint(150, 220, count)
        elif mode == 5:  # Arc
            colors[:, 0] = 200
            colors[:, 1] = 230
            colors[:, 2] = 255
            
        sizes = np.random.uniform(3.0, 5.5, count)
        lifes = np.random.uniform(0.6, 1.0, count)
        decays = np.random.uniform(0.8, 1.4, count)
        
        self.spawn_particles(count, pos[0], pos[1], vx, vy, colors, sizes, lifes, decays, attrib_type=3)
            
    def spawn_particles(self, num, pos_x, pos_y, vx, vy, color, size, life, decay, attrib_type=0):
        inactive = np.where(~self.active)[0]
        if len(inactive) == 0:
            return
        count = min(num, len(inactive))
        idx = inactive[:count]
        
        self.pos[idx, 0] = pos_x[:count] if isinstance(pos_x, np.ndarray) else pos_x
        self.pos[idx, 1] = pos_y[:count] if isinstance(pos_y, np.ndarray) else pos_y
        self.vel[idx, 0] = vx[:count] if isinstance(vx, np.ndarray) else vx
        self.vel[idx, 1] = vy[:count] if isinstance(vy, np.ndarray) else vy
        self.color[idx] = color[:count] if isinstance(color, np.ndarray) else color
        self.size[idx] = size[:count] if isinstance(size, np.ndarray) else size
        self.life[idx] = life[:count] if isinstance(life, np.ndarray) else life
        self.decay[idx] = decay[:count] if isinstance(decay, np.ndarray) else decay
        self.alpha[idx] = 255.0
        self.mode_attrib[idx, 0] = attrib_type
        self.mode_attrib[idx, 1] = np.random.uniform(0, 2 * math.pi, count)
        self.prev_pos[idx] = self.pos[idx].copy()
        self.active[idx] = True

    def update(self, gestures, dt):
        if dt <= 0:
            dt = 1.0/60.0
        
        # Accumulate elapsed time (for smooth noise fields)
        self._time += dt
            
        # Update charge timers for active hands based on current active gesture
        for side in ['Left', 'Right']:
            hand = gestures[side]
            if hand['active']:
                # Detect if any pinch is active (index, middle, or ring)
                is_pinching = (hand['pinch_active']['index'] or 
                               hand['pinch_active']['middle'] or 
                               hand['pinch_active']['ring'])
                               
                # Or fingers spread in Arc mode (Chain Lightning)
                is_spreading = False
                if self.mode == 5:
                    landmarks = hand['landmarks']
                    hand_scale = max(1.0, np.linalg.norm(landmarks[9] - landmarks[0]))
                    spread_dist = np.linalg.norm(landmarks[8][:2] - landmarks[12][:2]) / hand_scale
                    is_spreading = spread_dist > 0.65
                
                if is_pinching or is_spreading:
                    # Accumulate charge (up to max 3.0 seconds)
                    self.charge_time[side] = min(3.0, self.charge_time[side] + dt)
                else:
                    self.charge_time[side] = max(0.0, self.charge_time[side] - dt * 2.0)  # Decay charge if released without flick
            else:
                self.charge_time[side] = 0.0
            
        # 1. Mode transitions (dissolve)
        if self.transition_timer > 0.0:
            self.transition_timer -= dt
            # Slowly fade out all active particles
            active_mask = self.active
            self.alpha[active_mask] = np.clip(self.alpha[active_mask] - (255.0 / self.transition_duration) * dt, 0, 255)
            self.life[active_mask] -= dt * 2.0
            
            # Deactivate dead ones
            self.active = self.active & (self.life > 0)
            
            if self.transition_timer <= 0.0:
                self.reset_mode_particles(self.next_mode)
                # Mode switch burst effect!
                for side in ['Left', 'Right']:
                    hand = gestures[side]
                    if hand['active']:
                        self._trigger_mode_switch_burst(hand['palm_center'], self.mode)
                self.next_mode = None
            return # Skip normal updates during transition

        # Grab active mask
        active_mask = self.active

        # Decay lifetimes
        self.life[active_mask] -= self.decay[active_mask] * dt
        self.active = self.active & (self.life > 0)
        active_mask = self.active
        
        # Apply Drag (Base)
        drag = 0.98
        if self.mode == 4:  # CURRENT (Viscous liquid)
            drag = 0.88  # Slightly less aggressive drag for smoother momentum
        elif self.mode == 2:  # INFERNO
            drag = 0.94
        
        self.vel[active_mask] *= (drag ** (dt * 60.0))

        # 2. COORDINATE SUPER ABILITIES STATE MACHINE
        # B. Void Super (Cosmic Clap)
        if self.mode == 3 and gestures['clap_singularity'] and not self.super_active:
            self.super_active = True
            self.super_type = 3
            self.super_pos = gestures['clap_pos']
            self.super_timer = 1.0  # 1.0s (0.6s implosion, 0.4s blast)
            self.super_stage = 0    # 0: Implosion
            
        # B2. Void Super alternate trigger — Gojo Infinity Hand Sign
        elif self.mode == 3 and gestures['gojo_infinity'] and not self.super_active:
            # Trigger singularity at midpoint between the hands
            left_pos  = gestures['Left']['palm_center']
            right_pos = gestures['Right']['palm_center']
            self.super_active = True
            self.super_type = 3
            self.super_pos = (left_pos + right_pos) * 0.5
            self.super_timer = 1.0
            self.super_stage = 0
            
        # C. Inferno Super (Pyroclastic Sun) — now triggered by Torch Raise (index+middle up, both hands)
        elif self.mode == 2 and gestures['torch_raise'] and not self.super_active:
            self.super_active = True
            self.super_type = 2
            self.super_pos = gestures['torch_pos']
            self.super_timer = 1.2  # 1.2s charge
            self.super_stage = 0    # 0: Charge
            self.super_charge = 0.0
            
        # D. Current Super (Tsunami Sweep)
        elif self.mode == 4 and gestures['parallel_sweep'] and not self.super_active:
            self.super_active = True
            self.super_type = 4
            self.super_pos = gestures['Left']['palm_center'] if gestures['Left']['active'] else gestures['Right']['palm_center']
            self.super_vel = gestures['parallel_sweep_dir']
            self.super_timer = 1.5
            self.super_stage = 0
            # Spawn Tsunami wave!
            self._trigger_tsunami(self.super_vel)
            
        # E. Arc Super (Storm Core)
        elif self.mode == 5 and gestures['wrists_close_spread'] and not self.super_active:
            self.super_active = True
            self.super_type = 5
            self.super_pos = (gestures['Left']['wrist'] + gestures['Right']['wrist']) * 0.5
            self.super_timer = 1.8
            self.super_stage = 0

        # Update running Super abilities
        if self.super_active:
            self.super_timer -= dt
            
            # Void Super (Singularity) update
            if self.super_type == 3:
                # Sync legacy variables for main.py space warp compatibility
                self.singularity_active = True
                self.singularity_pos = self.super_pos
                self.singularity_stage = self.super_stage
                self.singularity_timer = self.super_timer
                
                if self.super_stage == 0 and self.super_timer <= 0.4:
                    # Transition to Detonation!
                    self.super_stage = 1
                    self.singularity_stage = 1
                    self._trigger_void_detonation()
                    
            # Inferno Super (Pyroclastic Sun) update
            elif self.super_type == 2:
                # Cancel if user stops bowl gesture
                if not gestures['bowl_shape'] and self.super_stage == 0:
                    self.super_active = False
                else:
                    self.super_pos = gestures['bowl_pos']
                    if self.super_stage == 0:
                        self._update_pyroclastic_sun_charge(dt)
                        if self.super_timer <= 0.0:
                            # Detonate!
                            self.super_stage = 1
                            self.super_timer = 1.0  # 1.0s blast fade
                            self._trigger_pyroclastic_sun_detonation()
                            
            # Arc Super (Storm Core) update
            elif self.super_type == 5:
                if not gestures['wrists_close_spread']:
                    self.super_active = False
                else:
                    self.super_pos = (gestures['Left']['wrist'] + gestures['Right']['wrist']) * 0.5
                    self._update_storm_core(dt)
                    
            # Deactivate when super finishes
            if self.super_timer <= 0.0 and self.super_stage != 0:
                self.super_active = False
                self.singularity_active = False

        # Apply forces based on active mode
        if self.mode == 1:
            self._update_surge(gestures, active_mask, dt)
        elif self.mode == 2:
            self._update_inferno(gestures, active_mask, dt)
        elif self.mode == 3:
            self._update_void(gestures, active_mask, dt)
        elif self.mode == 4:
            self._update_current(gestures, active_mask, dt)
        elif self.mode == 5:
            self._update_arc(gestures, active_mask, dt)

        # Update positions and store prev_pos for motion blur
        self.prev_pos[self.active] = self.pos[self.active].copy()
        self.pos[self.active] += self.vel[self.active] * dt

        # Screen boundary checks & wrapping/respawning
        active_indices = np.where(self.active)[0]
        if len(active_indices) > 0:
            xs = self.pos[active_indices, 0]
            ys = self.pos[active_indices, 1]
            types = self.mode_attrib[active_indices, 0]
            
            # Out of bounds check
            out_x = (xs < 0) | (xs > self.width)
            out_y = (ys < 0) | (ys > self.height)
            oob = out_x | out_y
            
            if np.any(oob):
                oob_indices = active_indices[oob]
                oob_types = types[oob]
                
                # Check for detonating launched special projectiles (type 1)
                launched_mask = oob_types == 1.0
                if np.any(launched_mask):
                    detonate_indices = oob_indices[launched_mask]
                    for idx in detonate_indices:
                        if self.mode == 1:
                            self._trigger_impact_explosion(self.pos[idx])
                        elif self.mode == 2:
                            self._trigger_inferno_explosion(self.pos[idx])
                        elif self.mode == 3:
                            self._trigger_void_rift_collapse(self.pos[idx])
                
                # Deactivate oob particles
                self.active[oob_indices] = False

    def _trigger_impact_explosion(self, impact_pos):
        self.shake_trigger = max(self.shake_trigger, 10.0)
        # Spawns a radial burst of particles at impact_pos
        count = min(35, np.sum(~self.active))
        if count == 0:
            return
        angles = np.random.uniform(0, 2*math.pi, count)
        speeds = np.random.uniform(200, 650, count)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = np.random.randint(0, 50, count)    # R
        colors[:, 1] = np.random.randint(180, 230, count)  # G
        colors[:, 2] = 255                                # B (Electric blue-white #00d4ff)
        
        self.spawn_particles(
            count,
            impact_pos[0], impact_pos[1],
            vx, vy,
            colors,
            np.random.uniform(2.0, 4.0, count),
            np.random.uniform(0.5, 0.9, count),
            np.random.uniform(0.8, 1.4, count),
            attrib_type=3
        )

    def _trigger_tsunami(self, dir_vec):
        self.shake_trigger = max(self.shake_trigger, 14.0)
        # Determine number of particles to spawn
        count = min(1200, np.sum(~self.active))
        if count == 0:
            return
            
        # Determine border and direction
        dir_norm = dir_vec / (np.linalg.norm(dir_vec) + 1e-5)
        
        px = np.zeros(count, dtype=np.float32)
        py = np.zeros(count, dtype=np.float32)
        vx = np.zeros(count, dtype=np.float32)
        vy = np.zeros(count, dtype=np.float32)
        
        # Horizontal sweep dominant
        if abs(dir_norm[0]) > abs(dir_norm[1]):
            if dir_norm[0] > 0:  # Sweeping Right, spawn on left edge (x=0)
                px.fill(0.0)
                py = np.random.uniform(0, self.height, count)
                vx = np.random.uniform(700, 1400, count)
                vy = np.random.uniform(-100, 100, count)
            else:  # Sweeping Left, spawn on right edge (x=width)
                px.fill(self.width)
                py = np.random.uniform(0, self.height, count)
                vx = np.random.uniform(-1400, -700, count)
                vy = np.random.uniform(-100, 100, count)
        # Vertical sweep dominant
        else:
            if dir_norm[1] > 0:  # Sweeping Down, spawn on top edge (y=0)
                px = np.random.uniform(0, self.width, count)
                py.fill(0.0)
                vx = np.random.uniform(-100, 100, count)
                vy = np.random.uniform(700, 1400, count)
            else:  # Sweeping Up, spawn on bottom edge (y=height)
                px = np.random.uniform(0, self.width, count)
                py.fill(self.height)
                vx = np.random.uniform(-100, 100, count)
                vy = np.random.uniform(-1400, -700, count)
                
        # Green-teal bioluminescent colors
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = 0
        colors[:, 1] = np.random.randint(200, 255, count) # G
        colors[:, 2] = np.random.randint(130, 200, count) # B
        
        sizes = np.random.uniform(3.0, 7.0, count)
        lifes = np.random.uniform(1.2, 1.8, count)
        decays = np.random.uniform(0.5, 0.9, count)
        
        self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=3)

    def _trigger_void_detonation(self):
        self.shake_trigger = max(self.shake_trigger, 24.0)
        count = min(1200, np.sum(~self.active))
        if count == 0:
            return
        angles = np.random.uniform(0, 2*math.pi, count)
        speeds = np.random.uniform(400, 1400, count)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        
        # Deep purple/white-hot colors
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = np.random.randint(160, 255, count) # R
        colors[:, 1] = np.random.randint(40, 255, count)  # G
        colors[:, 2] = 255                                # B
        
        sizes = np.random.uniform(3.0, 7.0, count)
        lifes = np.random.uniform(1.0, 2.2, count)
        decays = np.random.uniform(0.4, 0.8, count)
        
        self.spawn_particles(count, self.super_pos[0], self.super_pos[1], vx, vy, colors, sizes, lifes, decays, attrib_type=3)

    def _update_pyroclastic_sun_charge(self, dt):
        # Spawn swirling fire embers spiraling into sun core
        count = min(20, np.sum(~self.active))
        if count > 0:
            angles = np.random.uniform(0, 2*math.pi, count)
            radii = np.random.uniform(140, 280, count)
            px = self.super_pos[0] + np.cos(angles) * radii
            py = self.super_pos[1] + np.sin(angles) * radii
            
            # Velocity towards center + spiral rotation
            vx = -np.cos(angles) * 300.0 + np.sin(angles) * 150.0
            vy = -np.sin(angles) * 300.0 - np.cos(angles) * 150.0
            
            colors = np.zeros((count, 3), dtype=np.uint8)
            colors[:, 0] = 255
            colors[:, 1] = np.random.randint(180, 245, count) # Yellow
            colors[:, 2] = np.random.randint(0, 50, count)
            
            sizes = np.random.uniform(3.0, 6.5, count)
            lifes = np.random.uniform(0.7, 1.2, count)
            decays = np.random.uniform(0.6, 1.0, count)
            
            self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
            
        # Draw nearby active fire particles towards core
        active_mask = self.active
        d = self.super_pos - self.pos
        dist = np.linalg.norm(d, axis=1)
        suck_mask = (dist < 380.0) & active_mask & (self.mode_attrib[:, 0] == 0)
        
        if np.any(suck_mask):
            d_suck = d[suck_mask]
            dist_suck = dist[suck_mask][:, None]
            u_rad = d_suck / (dist_suck + 1e-5)
            self.vel[suck_mask] += u_rad * 800.0 * dt
            self.color[suck_mask] = [255, 230, 0] # Hot yellow-white

    def _trigger_pyroclastic_sun_detonation(self):
        self.shake_trigger = max(self.shake_trigger, 22.0)
        # Spawns concentric rings of fire expanding outwards
        for r_idx in range(4):
            speed = 280.0 * (r_idx + 1)
            count = 200
            angles = np.linspace(0, 2*math.pi, count, endpoint=False) + random.uniform(0, 0.1)
            vx = np.cos(angles) * speed
            vy = np.sin(angles) * speed
            
            colors = np.zeros((count, 3), dtype=np.uint8)
            if r_idx == 3: # Outermost ring: red
                colors[:, 0] = np.random.randint(180, 220, count)
                colors[:, 1] = 0
                colors[:, 2] = 0
            elif r_idx == 2: # Orange
                colors[:, 0] = 255
                colors[:, 1] = np.random.randint(80, 130, count)
                colors[:, 2] = 0
            else: # Inner rings: yellow/white-hot
                colors[:, 0] = 255
                colors[:, 1] = np.random.randint(200, 255, count)
                colors[:, 2] = np.random.randint(100, 200, count)
                
            sizes = np.random.uniform(4.5, 9.0, count) - r_idx * 0.8
            lifes = np.random.uniform(1.2, 2.0, count)
            decays = np.random.uniform(0.4, 0.7, count)
            
            self.spawn_particles(count, self.super_pos[0], self.super_pos[1], vx, vy, colors, sizes, lifes, decays, attrib_type=3)

    def _update_storm_core(self, dt):
        self.shake_trigger = max(self.shake_trigger, 6.0)
        # Spawn volatile electrical spark particles at core
        count = min(15, np.sum(~self.active))
        if count > 0:
            angles = np.random.uniform(0, 2*math.pi, count)
            speeds = np.random.uniform(150, 450, count)
            vx = np.cos(angles) * speeds
            vy = np.sin(angles) * speeds
            
            colors = np.zeros((count, 3), dtype=np.uint8)
            colors[:, 0] = np.random.randint(180, 230, count)
            colors[:, 1] = np.random.randint(220, 255, count)
            colors[:, 2] = 255
            
            sizes = np.random.uniform(2.0, 4.5, count)
            lifes = np.random.uniform(0.3, 0.8, count)
            decays = np.random.uniform(1.2, 2.0, count)
            
            self.spawn_particles(count, self.super_pos[0], self.super_pos[1], vx, vy, colors, sizes, lifes, decays, attrib_type=0)
            
        # Draw electric sparks rapidly towards core
        active_mask = self.active
        d = self.super_pos - self.pos
        dist = np.linalg.norm(d, axis=1)
        att_mask = (dist < 320.0) & active_mask
        
        if np.any(att_mask):
            d_att = d[att_mask]
            dist_att = dist[att_mask][:, None]
            u_dir = d_att / (dist_att + 1e-5)
            # High volatile acceleration + random jitter
            self.vel[att_mask] += u_dir * 1000.0 * dt + np.random.uniform(-400, 400, (np.sum(att_mask), 2)) * dt

    def _trigger_inferno_explosion(self, pos):
        self.shake_trigger = max(self.shake_trigger, 12.0)
        count = min(50, np.sum(~self.active))
        if count == 0:
            return
        angles = np.random.uniform(0, 2*math.pi, count)
        speeds = np.random.uniform(200, 600, count)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = 255
        colors[:, 1] = np.random.randint(50, 150, count)
        colors[:, 2] = 0
        
        self.spawn_particles(
            count, pos[0], pos[1], vx, vy, colors,
            np.random.uniform(3.0, 6.0, count),
            np.random.uniform(0.5, 1.0, count),
            np.random.uniform(0.8, 1.5, count),
            attrib_type=3
        )

    def _trigger_void_rift_collapse(self, pos):
        self.shake_trigger = max(self.shake_trigger, 8.0)
        count = min(50, np.sum(~self.active))
        if count == 0:
            return
        angles = np.random.uniform(0, 2*math.pi, count)
        speeds = np.random.uniform(-400, -150, count) # Inward velocities
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = 138
        colors[:, 1] = 43
        colors[:, 2] = 226
        
        self.spawn_particles(
            count, pos[0] + np.cos(angles)*120.0, pos[1] + np.sin(angles)*120.0, vx, vy, colors,
            np.random.uniform(2.5, 4.5, count),
            np.random.uniform(0.4, 0.8, count),
            np.random.uniform(1.2, 1.8, count),
            attrib_type=3
        )

    def _update_surge(self, gestures, active_mask, dt):
        # Check if both hands are active and pinching
        left_hand = gestures['Left']
        right_hand = gestures['Right']
        both_pinching = (left_hand['active'] and right_hand['active'] and 
                         left_hand['pinch_active']['index'] and right_hand['pinch_active']['index'])
                         
        if both_pinching:
            # 1. DUAL-HAND PLASMA CORE (EXPANDS AND SHOOTS OUT)
            p1 = (left_hand['landmarks'][4][:2] + left_hand['landmarks'][8][:2]) * 0.5
            p2 = (right_hand['landmarks'][4][:2] + right_hand['landmarks'][8][:2]) * 0.5
            mid_center = (p1 + p2) * 0.5
            hand_dist = np.linalg.norm(p1 - p2)
            
            # Combined charge from both hands
            avg_charge = (self.charge_time['Left'] + self.charge_time['Right']) * 0.5
            charge_mult = 1.0 + avg_charge * 1.5  # 1x -> 5.5x over 3 seconds
            
            # Spawn core particles at center — more with charge
            if np.sum(self.active) < self.max_particles:
                count = max(7, int(7 * charge_mult))
                px = np.random.uniform(mid_center[0] - 8, mid_center[0] + 8, count)
                py = np.random.uniform(mid_center[1] - 8, mid_center[1] + 8, count)
                
                # Expand core: orbit speed and radius proportional to hand_dist + charge
                angles = np.random.uniform(0, 2*math.pi, count)
                orbit_spd = 120.0 * charge_mult
                vx = np.cos(angles) * orbit_spd
                vy = np.sin(angles) * orbit_spd
                
                colors = np.zeros((count, 3), dtype=np.uint8)
                # Electric blue shades; white-hot when fully charged
                colors[:, 0] = min(255, int(avg_charge / 3.0 * 200))
                colors[:, 1] = np.random.randint(180, 230, count)
                colors[:, 2] = 255
                
                sizes = np.random.uniform(2.5, 5.0 + avg_charge * 1.5, count)
                lifes = np.random.uniform(1.5, 2.5, count)
                decays = np.random.uniform(0.1, 0.18, count)
                self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                
            # Physics: orbit around mid_center with radius = hand_dist * 0.5
            d = mid_center - self.pos
            dist = np.linalg.norm(d, axis=1)
            
            # Pull active normal particles into the dual-hand orbit field
            orbit_field = (dist < hand_dist * 0.8) & active_mask & (self.mode_attrib[:, 0] == 0)
            if np.any(orbit_field):
                d_orb = d[orbit_field]
                dist_orb = dist[orbit_field][:, None]
                u_rad = d_orb / (dist_orb + 1e-5)
                u_tan = np.zeros_like(u_rad)
                u_tan[:, 0] = -u_rad[:, 1]
                u_tan[:, 1] = u_rad[:, 0]
                
                # Orbit and spring pull to hand_dist * 0.5 boundary
                target_r = hand_dist * 0.5
                radial_force = (dist_orb - target_r) * 6.0
                self.vel[orbit_field] = u_tan * 420.0 + u_rad * radial_force
                self.size[orbit_field] = np.clip(1.5 + hand_dist * 0.008, 1.5, 6.5)
                
            # Action: flick either hand to trigger a massive SUPERNOVA BLAST (shoots out blue particles)
            if left_hand['flick'] or right_hand['flick']:
                # Scale blast by combined charge: 320 particles at 0s, up to 800 at 3s
                charge_blast = (self.charge_time['Left'] + self.charge_time['Right']) * 0.5
                blast_shake = 20.0 + charge_blast * 10.0
                self.shake_trigger = max(self.shake_trigger, blast_shake)
                base_count = int(320 + charge_blast * 160)
                count = min(base_count, np.sum(~self.active))
                if count > 0:
                    angles = np.random.uniform(0, 2*math.pi, count)
                    speed_min = 500.0 + charge_blast * 300.0
                    speed_max = 1400.0 + charge_blast * 600.0
                    speeds = np.random.uniform(speed_min, speed_max, count)
                    vx = np.cos(angles) * speeds
                    vy = np.sin(angles) * speeds
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    # White-hot core particles when fully charged
                    colors[:, 0] = min(255, int(charge_blast / 3.0 * 255))
                    colors[:, 1] = np.random.randint(180, 235, count)
                    colors[:, 2] = 255
                    
                    size_max = 6.5 + charge_blast * 2.5
                    self.spawn_particles(
                        count, mid_center[0], mid_center[1], vx, vy, colors,
                        np.random.uniform(3.5, size_max, count),
                        np.random.uniform(1.2, 2.5 + charge_blast * 0.5, count),
                        np.random.uniform(0.3, 0.7, count),
                        attrib_type=3 # Explosion shard
                    )
                    # Reset charge on release
                    self.charge_time['Left'] = 0.0
                    self.charge_time['Right'] = 0.0
                    
        else:
            # 2. SINGLE-HAND POSTURES (Surge attraction, single pinch grab/launch)
            for side in ['Left', 'Right']:
                hand = gestures[side]
                if not hand['active']:
                    continue
                
                pinch_active = hand['pinch_active']['index']
                pinch_dist = hand['pinch_dists']['index']
                palm_pos = hand['palm_center']
                
                # Check for open hand and spawn outward shooting particles
                avg_curl = np.mean(hand['curls'][1:]) # index, middle, ring, pinky
                if avg_curl < 0.35:
                    if np.sum(self.active) < self.max_particles:
                        count = 2
                        px = np.random.uniform(palm_pos[0] - 10, palm_pos[0] + 10, count)
                        py = np.random.uniform(palm_pos[1] - 10, palm_pos[1] + 10, count)
                        angles = np.random.uniform(0, 2*math.pi, count)
                        speeds = np.random.uniform(150, 400, count)
                        vx = np.cos(angles) * speeds
                        vy = np.sin(angles) * speeds
                        colors = np.zeros((count, 3), dtype=np.uint8)
                        colors[:] = [0, 212, 255]
                        sizes = np.random.uniform(1.5, 3.2, count)
                        lifes = np.random.uniform(0.6, 1.0, count)
                        decays = np.random.uniform(0.8, 1.4, count)
                        self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                    
                    # SUPER: Open-palm FAST FLICK → Supernova (no double pinch needed!)
                    if hand['flick']:
                        ch = self.charge_time[side]
                        self._trigger_surge_supernova(palm_pos, ch)
                        self.charge_time[side] = 0.0

                if pinch_active:
                    pinch_center = (hand['landmarks'][4][:2] + hand['landmarks'][8][:2]) * 0.5
                    
                    # Spawn new particles at pinch center to form the "sphere of light"
                    # Scale count and speed by charge time
                    if np.sum(self.active) < self.max_particles:
                        ch = self.charge_time[side]
                        count = max(4, int(4 + ch * 4))  # 4 -> 16 particles
                        px = np.random.uniform(pinch_center[0] - 8, pinch_center[0] + 8, count)
                        py = np.random.uniform(pinch_center[1] - 8, pinch_center[1] + 8, count)
                        angles = np.random.uniform(0, 2*math.pi, count)
                        spd_min = 80.0 + ch * 60.0
                        spd_max = 220.0 + ch * 120.0
                        speeds = np.random.uniform(spd_min, spd_max, count)
                        vx = np.cos(angles) * speeds
                        vy = np.sin(angles) * speeds
                        colors = np.zeros((count, 3), dtype=np.uint8)
                        # Shift toward white-hot as charge fills
                        r_val = min(255, int(ch / 3.0 * 220))
                        colors[:, 0] = r_val
                        colors[:, 1] = min(255, 212 + int(ch / 3.0 * 43))
                        colors[:, 2] = 255
                        sizes = np.random.uniform(1.5, 3.5 + ch * 1.5, count)
                        lifes = np.random.uniform(1.2, 2.0, count)
                        decays = np.random.uniform(0.1, 0.25, count)
                        self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                    
                    # Vector from particles to pinch center
                    d = pinch_center - self.pos
                    dist = np.linalg.norm(d, axis=1)
                    
                    # Grab nearby particles (within 160 pixels)
                    near_mask = (dist < 160.0) & active_mask & (self.mode_attrib[:, 0] == 0)
                    
                    # Setup orbit physics
                    if np.any(near_mask):
                        d_near = d[near_mask]
                        dist_near = dist[near_mask][:, None]
                        
                        target_r = max(10.0, pinch_dist * 350.0)
                        u_rad = d_near / (dist_near + 1e-5)
                        u_tan = np.zeros_like(u_rad)
                        u_tan[:, 0] = -u_rad[:, 1]
                        u_tan[:, 1] = u_rad[:, 0]
                        
                        orbit_speed = 300.0 + hand['curls'][2] * 400.0
                        rot_delta = hand['rotation'] - self.prev_rotation[side]
                        self.prev_rotation[side] = hand['rotation']
                        
                        radial_force = (dist_near - target_r) * 8.0
                        vel_orbit = u_tan * orbit_speed + u_rad * radial_force
                        
                        if abs(rot_delta) > 0.005:
                            vel_orbit += u_tan * (rot_delta * 4000.0)
                            
                        self.vel[near_mask] = vel_orbit
                        self.color[near_mask] = [255, 255, 255] if random.random() < 0.25 else [0, 212, 255]
                        self.size[near_mask] = np.clip(5.0 - pinch_dist * 12.0, 1.5, 6.0)
                    
                    # Check for fast wrist flick to LAUNCH the grabbed sphere
                    if hand['flick']:
                        launch_mask = (dist < 180.0) & active_mask
                        if np.any(launch_mask):
                            ch = self.charge_time[side]
                            flick_dir = hand['flick_dir']
                            self.mode_attrib[launch_mask, 0] = 1.0
                            # Charge scales launch speed 1500–3500 px/s
                            launch_spd_min = 1500.0 + ch * 500.0
                            launch_spd_max = 2000.0 + ch * 500.0
                            self.vel[launch_mask] = flick_dir * np.random.uniform(launch_spd_min, launch_spd_max, np.sum(launch_mask))[:, None]
                            self.decay[launch_mask] = max(0.02, 0.05 - ch * 0.01)  # Slower decay when charged
                            self.color[launch_mask] = [255, 255, 255]
                            self.shake_trigger = max(self.shake_trigger, 8.0 + ch * 6.0)
                            self.charge_time[side] = 0.0  # Reset charge on launch
                else:
                    d = palm_pos - self.pos
                    dist = np.linalg.norm(d, axis=1)
                    inf_mask = (dist < 350.0) & active_mask & (self.mode_attrib[:, 0] == 0)
                    
                    if np.any(inf_mask):
                        d_inf = d[inf_mask]
                        dist_inf = dist[inf_mask][:, None]
                        dir_inf = d_inf / (dist_inf + 1e-5)
                        
                        if avg_curl < 0.35:
                            force_strength = -400.0 * (1.0 - (dist_inf / 350.0))
                        elif avg_curl > 0.7:
                            force_strength = 600.0 * (1.0 - (dist_inf / 350.0))
                        else:
                            force_strength = 0.0
                            
                        self.vel[inf_mask] += dir_inf * force_strength * dt
                    
        # Maintain ambient counts (unused)
        pass

    # ─────────── SURGE single-hand open-palm SUPERNOVA ───────────
    def _trigger_surge_supernova(self, origin, charge):
        """Single-hand Supernova: open-palm fast flick while in Surge mode."""
        count = min(int(220 + charge * 120), np.sum(~self.active))
        if count == 0:
            return
        self.shake_trigger = max(self.shake_trigger, 18.0 + charge * 8.0)
        angles = np.random.uniform(0, 2 * math.pi, count)
        spd_min = 450.0 + charge * 250.0
        spd_max = 1200.0 + charge * 600.0
        speeds = np.random.uniform(spd_min, spd_max, count)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        colors = np.zeros((count, 3), dtype=np.uint8)
        colors[:, 0] = min(255, int(charge / 3.0 * 220))
        colors[:, 1] = np.random.randint(180, 235, count)
        colors[:, 2] = 255
        self.spawn_particles(
            count, origin[0], origin[1], vx, vy, colors,
            np.random.uniform(3.0, 7.0 + charge * 1.5, count),
            np.random.uniform(1.0, 2.5 + charge * 0.4, count),
            np.random.uniform(0.3, 0.7, count),
            attrib_type=3
        )
        
    def _update_inferno(self, gestures, active_mask, dt):
        # Inferno: fire embers rising, open palm columns, fireball launching (special), trails, fist snuffing smoke
        
        # Apply upward buoyancy to all active particles
        # Normal flame particles rise, smoke (type 2) rises slower and drifts
        normal_flame = active_mask & (self.mode_attrib[:, 0] == 0)
        smoke_mask = active_mask & (self.mode_attrib[:, 0] == 2)
        projectiles = active_mask & (self.mode_attrib[:, 0] == 1)
        
        self.vel[normal_flame, 1] -= 320.0 * dt  # Strong upward rise
        self.vel[smoke_mask, 1] -= 120.0 * dt    # Slower smoke rise
        
        # Projectiles: leave a trail of smoke and sparks as they travel
        if np.any(projectiles):
            proj_indices = np.where(projectiles)[0]
            for idx in proj_indices:
                if random.random() < 0.35:
                    # Spawn spark trail
                    self.spawn_particles(
                        1, self.pos[idx, 0], self.pos[idx, 1],
                        np.random.uniform(-50, 50), np.random.uniform(-50, 50),
                        [255, 100, 0], np.random.uniform(2.5, 4.5),
                        np.random.uniform(0.4, 0.7), np.random.uniform(1.2, 1.8),
                        attrib_type=0
                    )
                if random.random() < 0.20:
                    # Spawn smoke trail
                    self.spawn_particles(
                        1, self.pos[idx, 0], self.pos[idx, 1],
                        np.random.uniform(-30, 30), np.random.uniform(-60, -20),
                        [90, 90, 90], np.random.uniform(4.0, 7.0),
                        np.random.uniform(0.3, 0.6), np.random.uniform(1.5, 2.2),
                        attrib_type=2
                    )

        # Color progression for flames: White-hot -> yellow -> orange -> red -> fade
        if np.any(normal_flame):
            life_flame = self.life[normal_flame]
            colors = np.zeros((np.sum(normal_flame), 3), dtype=np.uint8)
            
            white_mask = life_flame > 0.8
            yellow_mask = (life_flame <= 0.8) & (life_flame > 0.5)
            orange_mask = (life_flame <= 0.5) & (life_flame > 0.2)
            red_mask = life_flame <= 0.2
            
            colors[white_mask] = [255, 245, 230] # White-hot
            colors[yellow_mask] = [255, 215, 0]   # Yellow
            colors[orange_mask] = [255, 100, 0]   # Orange
            colors[red_mask] = [180, 20, 0]       # Red
            
            self.color[normal_flame] = colors
            
        # Particle spawning logic at hands
        for side in ['Left', 'Right']:
            hand = gestures[side]
            if not hand['active']:
                continue
            
            palm_pos = hand['palm_center']
            wrist_pos = hand['wrist']
            
            avg_curl = np.mean(hand['curls'][1:])
            
            # Detect fist (snuffing flames into smoke)
            if avg_curl > 0.82:
                # Find flame particles close to fist and turn them to smoke (type 2)
                d = palm_pos - self.pos
                dist = np.linalg.norm(d, axis=1)
                snuff_mask = (dist < 110.0) & normal_flame
                
                if np.any(snuff_mask):
                    self.mode_attrib[snuff_mask, 0] = 2.0  # Turn to smoke
                    self.color[snuff_mask] = np.random.randint(70, 110, (np.sum(snuff_mask), 3)).astype(np.uint8)
                    self.size[snuff_mask] = self.size[snuff_mask] * 1.5 # Smoke swells
                    self.decay[snuff_mask] = 0.5 # Fades faster
                continue # Do not spawn fire if fist is closed
            
            # Special Ability: Fireball Launcher — INDEX GUN shape + flick
            # Gun shape: index extended (curl < 0.30), middle/ring/pinky curled (> 0.60)
            index_curl  = hand['curls'][1]
            mid_curl    = hand['curls'][2]
            ring_curl   = hand['curls'][3]
            pinky_curl  = hand['curls'][4]
            gun_shape = (index_curl < 0.32 and mid_curl > 0.58 and ring_curl > 0.58)
            if gun_shape:
                # Index fingertip position = muzzle
                muzzle = hand['landmarks'][8][:2]
                ch = self.charge_time[side]
                
                # Condense fire at muzzle — scales with charge
                if np.sum(self.active) < self.max_particles:
                    count = max(3, int(3 + ch * 7))  # 3 → 24 particles
                    px = np.random.uniform(muzzle[0] - 5, muzzle[0] + 5, count)
                    py = np.random.uniform(muzzle[1] - 5, muzzle[1] + 5, count)
                    angles = np.random.uniform(0, 2 * math.pi, count)
                    speeds = np.random.uniform(40 + ch * 30, 120 + ch * 70, count)
                    vx = np.cos(angles) * speeds
                    vy = np.sin(angles) * speeds
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    colors[:, 0] = 255
                    colors[:, 1] = min(255, 150 + int(ch / 3.0 * 105))
                    colors[:, 2] = min(255, int(ch / 3.0 * 180))
                    sizes = np.random.uniform(2.5 + ch * 0.5, 5.0 + ch * 3.0, count)
                    lifes = np.random.uniform(0.4, 0.8, count)
                    decays = np.random.uniform(0.3, 0.6, count)
                    self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                
                # Flick to shoot fireball!
                if hand['flick']:
                    flick_dir = hand['flick_dir']
                    fb_speed = 1700.0 + ch * 600.0
                    fb_size  = 12.0  + ch * 10.0
                    fb_life  = 2.0   + ch * 1.0
                    self.shake_trigger = max(self.shake_trigger, 8.0 + ch * 8.0)
                    self.spawn_particles(
                        1, muzzle[0], muzzle[1],
                        flick_dir[0] * fb_speed, flick_dir[1] * fb_speed,
                        [255, 230, 100], fb_size, fb_life, 0.05,
                        attrib_type=1
                    )
                    self.charge_time[side] = 0.0
                continue  # Skip normal flame spawn while gun is primed
                
            # Normal flame: single index finger pointing up = fire pillar from fingertip
            # Gun shape (index only extended): fire erupts from index tip upward
            is_index_up = (index_curl < 0.32 and mid_curl > 0.50 and ring_curl > 0.50)
            if is_index_up:
                # Fire pillar from index tip
                tip_idx = hand['landmarks'][8][:2]
                count = int(4 + (1.0 - index_curl) * 8)  # 4-12 particles
                px = np.random.uniform(tip_idx[0] - 8, tip_idx[0] + 8, count)
                py = np.random.uniform(tip_idx[1] - 8, tip_idx[1] + 8, count)
                vx = np.random.uniform(-35, 35, count)
                vy = np.random.uniform(-480, -220, count)  # Upward
                colors = np.zeros((count, 3), dtype=np.uint8)
                colors[:] = [255, 245, 230]  # White-hot
                sizes = np.random.uniform(3.5, 9.0, count)
                self.spawn_particles(
                    count, px, py, vx, vy, colors, sizes,
                    np.random.uniform(0.6, 1.1, count),
                    np.random.uniform(0.65, 1.05, count),
                    attrib_type=0
                )
            elif avg_curl < 0.4:  # Open palm fallback: ambient embers
                count = 2
                px = np.random.uniform(palm_pos[0] - 20, palm_pos[0] + 20, count)
                py = np.random.uniform(palm_pos[1] - 10, palm_pos[1] + 10, count)
                vx = np.random.uniform(-30, 30, count)
                vy = np.random.uniform(-200, -80, count)
                colors = np.zeros((count, 3), dtype=np.uint8)
                colors[:] = [255, 140, 0]
                self.spawn_particles(count, px, py, vx, vy, colors,
                                     np.random.uniform(2.5, 6.0, count),
                                     np.random.uniform(0.5, 1.0, count),
                                     np.random.uniform(0.8, 1.3, count), attrib_type=0)
                
            # Trail fire behind fast hand movement
            flick_dir = hand['flick_dir']
            if hand['flick'] and np.linalg.norm(flick_dir) > 0.1:
                count = 45
                px = np.random.uniform(palm_pos[0] - 15, palm_pos[0] + 15, count)
                py = np.random.uniform(palm_pos[1] - 15, palm_pos[1] + 15, count)
                
                vx = -flick_dir[0] * np.random.uniform(300, 600, count)
                vy = -flick_dir[1] * np.random.uniform(300, 600, count) - 100.0
                
                colors = np.zeros((count, 3), dtype=np.uint8)
                colors[:] = [255, 120, 0]
                
                self.spawn_particles(
                    count, px, py, vx, vy, colors,
                    np.random.uniform(3.0, 6.0, count),
                    np.random.uniform(0.4, 0.8, count),
                    np.random.uniform(1.2, 2.0, count),
                    attrib_type=0
                )

    def _update_void(self, gestures, active_mask, dt):
        # Void: Space warp, massive gravity pull towards fists, singularity implosion/explosion on clap, travelling Void Rifts (special)
        
        # 1. Singularity physics
        if self.singularity_active:
            s_pos = self.singularity_pos
            d = s_pos - self.pos
            dist = np.linalg.norm(d, axis=1)
            
            if self.singularity_stage == 0:  # Implosion (Spiral)
                # Spawn inward spiraling particles for implosion
                if np.sum(self.active) < self.max_particles:
                    count = 16
                    angles = np.random.uniform(0, 2*math.pi, count)
                    radii = np.random.uniform(120, 280, count)
                    px = s_pos[0] + np.cos(angles) * radii
                    py = s_pos[1] + np.sin(angles) * radii
                    vx = -np.cos(angles) * 150.0 + np.sin(angles) * 60.0
                    vy = -np.sin(angles) * 150.0 - np.cos(angles) * 60.0
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    colors[:] = [138, 43, 226]
                    sizes = np.random.uniform(2.5, 4.0, count)
                    lifes = np.random.uniform(0.8, 1.4, count)
                    decays = np.random.uniform(0.5, 1.0, count)
                    self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)

                # Particles spiral into singularity
                near_mask = active_mask & (dist > 5.0)
                if np.any(near_mask):
                    d_near = d[near_mask]
                    dist_near = dist[near_mask][:, None]
                    u_rad = d_near / dist_near
                    
                    # Tangential vector (spiral spin direction)
                    u_tan = np.zeros_like(u_rad)
                    u_tan[:, 0] = -u_rad[:, 1]
                    u_tan[:, 1] = u_rad[:, 0]
                    
                    # High gravity pull + orbital spin
                    pull_force = 1200.0 * (1.0 - np.clip(dist_near / 800.0, 0.0, 0.9))
                    spin_force = 800.0 * (1.0 - np.clip(dist_near / 800.0, 0.0, 0.9))
                    
                    self.vel[near_mask] = u_rad * pull_force + u_tan * spin_force
                    self.color[near_mask] = [138, 43, 226] # Deep purple
                    
            elif self.singularity_stage == 1:  # Detonation
                # Push existing particles outward quickly
                near_mask = active_mask & (dist > 1.0)
                if np.any(near_mask):
                    d_near = d[near_mask]
                    dist_near = dist[near_mask][:, None]
                    u_rad = -d_near / dist_near  # Repelling direction (away from center)
                    
                    # Radial outward blast
                    blast_speed = 1000.0 / (dist_near * 0.005 + 0.5)
                    self.vel[near_mask] = u_rad * blast_speed
                    
            return # Skip fist gravity if singularity clap is running

        # Update Void Rift Projectiles (apply localized gravity to normal particles)
        projectiles = active_mask & (self.mode_attrib[:, 0] == 1)
        if np.any(projectiles):
            proj_indices = np.where(projectiles)[0]
            normal_particles = active_mask & (self.mode_attrib[:, 0] == 0)
            if np.any(normal_particles):
                for p_idx in proj_indices:
                    p_pos = self.pos[p_idx]
                    d_proj = p_pos - self.pos
                    dist_proj = np.linalg.norm(d_proj, axis=1)
                    
                    # Pull normal particles within 180px towards projectile
                    pull_mask = (dist_proj < 180.0) & normal_particles
                    if np.any(pull_mask):
                        d_pull = d_proj[pull_mask]
                        dist_pull = dist_proj[pull_mask][:, None]
                        u_dir = d_pull / (dist_pull + 1e-5)
                        
                        # Soft pull force
                        pull_speed = 450.0 * (1.0 - (dist_pull / 180.0))
                        self.vel[pull_mask] += u_dir * pull_speed * dt
                        self.color[pull_mask] = [100, 20, 180] # Fade colors to purple
                        
                    # Spawn mini-dust trails
                    if random.random() < 0.25:
                        self.spawn_particles(
                            1, p_pos[0], p_pos[1],
                            np.random.uniform(-40, 40), np.random.uniform(-40, 40),
                            [138, 43, 226], np.random.uniform(2.0, 3.5),
                            np.random.uniform(0.3, 0.7), np.random.uniform(1.2, 2.0),
                            attrib_type=0
                        )

        # 2. Fist Gravity Wells
        for side in ['Left', 'Right']:
            hand = gestures[side]
            if not hand['active']:
                continue
            
            avg_curl = np.mean(hand['curls'][1:])
            
            # Special Ability: Void Rift — V-SIGN (index+middle extended, ring+pinky curled) + flick
            # Two-finger point = peace sign / V for void rift
            v_index_curl = hand['curls'][1]
            v_mid_curl   = hand['curls'][2]
            v_ring_curl  = hand['curls'][3]
            v_pinky_curl = hand['curls'][4]
            v_sign = (v_index_curl < 0.35 and v_mid_curl < 0.40 and
                      v_ring_curl  > 0.55 and v_pinky_curl > 0.55)
            if v_sign:
                # Use midpoint of index+middle tips as the rift focus
                tip_idx  = hand['landmarks'][8][:2]
                tip_mid  = hand['landmarks'][12][:2]
                rift_center = (tip_idx + tip_mid) * 0.5
                ch = self.charge_time[side]
                
                # Condense dark matter at the two-finger tip
                if np.sum(self.active) < self.max_particles:
                    count = max(4, int(4 + ch * 6))  # 4 → 22 particles
                    px = np.random.uniform(rift_center[0] - 8, rift_center[0] + 8, count)
                    py = np.random.uniform(rift_center[1] - 8, rift_center[1] + 8, count)
                    angles = np.random.uniform(0, 2 * math.pi, count)
                    speeds = np.random.uniform(60 + ch * 20, 180 + ch * 60, count)
                    vx = np.cos(angles) * speeds
                    vy = np.sin(angles) * speeds
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    colors[:, 0] = min(255, 100 + int(ch / 3.0 * 100))
                    colors[:, 1] = 20
                    colors[:, 2] = min(255, 180 + int(ch / 3.0 * 75))
                    sizes = np.random.uniform(2.5, 4.0 + ch * 2.0, count)
                    lifes  = np.random.uniform(0.8, 1.4, count)
                    decays = np.random.uniform(0.2, 0.4, count)
                    self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                
                if hand['flick']:
                    flick_dir = hand['flick_dir']
                    rift_speed = 1400.0 + ch * 500.0
                    rift_size  = 10.0  + ch * 8.0
                    rift_life  = 2.0   + ch * 1.5
                    self.shake_trigger = max(self.shake_trigger, 6.0 + ch * 6.0)
                    self.spawn_particles(
                        1, rift_center[0], rift_center[1],
                        flick_dir[0] * rift_speed, flick_dir[1] * rift_speed,
                        [147, 112, 219], rift_size, rift_life, 0.05,
                        attrib_type=1
                    )
                    self.charge_time[side] = 0.0
                continue  # Skip fist gravity when preparing rift
            
            # If closed fist, create a black hole gravitational well at palm center
            if avg_curl > 0.8:
                fist_pos = hand['palm_center']
                
                # Spawn incoming gravitational dust particles surrounding the fist
                if np.sum(self.active) < self.max_particles:
                    count = 5
                    angles = np.random.uniform(0, 2*math.pi, count)
                    radii = np.random.uniform(160, 300, count)
                    px = fist_pos[0] + np.cos(angles) * radii
                    py = fist_pos[1] + np.sin(angles) * radii
                    # Velocity towards center + spiral rotation
                    vx = -np.cos(angles) * 120.0 + np.sin(angles) * 40.0
                    vy = -np.sin(angles) * 120.0 - np.cos(angles) * 40.0
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    colors[:] = [138, 43, 226]  # Deep purple
                    sizes = np.random.uniform(2.0, 4.5, count)
                    lifes = np.random.uniform(1.0, 1.8, count)
                    decays = np.random.uniform(0.3, 0.6, count)
                    self.spawn_particles(count, px, py, vx, vy, colors, sizes, lifes, decays, attrib_type=0)
                d = fist_pos - self.pos
                dist = np.linalg.norm(d, axis=1)
                
                # Gravity influence radius: 450px
                grav_mask = (dist < 450.0) & active_mask
                if np.any(grav_mask):
                    d_grav = d[grav_mask]
                    dist_grav = dist[grav_mask][:, None]
                    u_dir = d_grav / (dist_grav + 1e-5)
                    
                    # Gravitational strength: inverse square-ish law
                    gravity_strength = 200000.0 / (dist_grav + 50.0)
                    self.vel[grav_mask] += u_dir * gravity_strength * dt
                    
                    # Change color of sucked particles to dark violet-black
                    self.color[grav_mask] = [75, 0, 130] # Violet

    def _update_current(self, gestures, active_mask, dt):
        """Current: viscous liquid flow — smooth curl-noise drift, hand parting, whirlpool."""
        
        n_active = np.sum(active_mask)
        if n_active > 0:
            px = self.pos[active_mask, 0]
            py = self.pos[active_mask, 1]
            t  = self._time

            # ---- Smooth curl-noise flow field ----
            # Curl of a potential field F(x,y,t) gives a divergence-free velocity.
            # We use two offset sine-wave potentials for organic swirling.
            freq  = 0.010   # spatial frequency (lower = broader waves)
            freq2 = 0.007
            amp   = 32.0    # base drift amplitude (px/s)

            # dF/dy  (x-component of curl)
            curl_x = ( amp * np.cos(freq  * py + t * 0.6 + np.sin(freq2 * px + t * 0.4) * 1.8)
                     + amp * 0.5 * np.cos(freq2 * py - t * 0.45 + 1.2) )
            # -dF/dx  (y-component of curl)
            curl_y = (-amp * np.cos(freq  * px + t * 0.55 + np.sin(freq2 * py + t * 0.35) * 1.8)
                     - amp * 0.5 * np.cos(freq2 * px + t * 0.5 + 2.1) )

            # Apply as a gentle push (not a hard assignment) so existing momentum blends
            self.vel[active_mask, 0] += curl_x * dt * 60.0
            self.vel[active_mask, 1] += curl_y * dt * 60.0

            # Gentle cohesion: nudge particles slightly toward the local centroid
            # (gives a blobby, cohesive liquid feeling)
            if n_active > 10:
                cx = float(np.mean(px))
                cy = float(np.mean(py))
                self.vel[active_mask, 0] += (cx - px) * 0.4 * dt
                self.vel[active_mask, 1] += (cy - py) * 0.4 * dt

        # Color waves: #00fa9a (spring green) → #00ced1 (dark turquoise) based on life
        if np.any(active_mask):
            lifes = self.life[active_mask]
            colors = np.zeros((np.sum(active_mask), 3), dtype=np.uint8)
            colors[:, 0] = 0
            colors[:, 1] = np.clip((lifes * 50.0 + 200.0), 0, 255).astype(np.uint8)
            colors[:, 2] = np.clip((lifes * 150.0 + 100.0), 0, 255).astype(np.uint8)
            self.color[active_mask] = colors
            
        # Parting around fingers & Special Whirlpool Vortex
        for side in ['Left', 'Right']:
            hand = gestures[side]
            if not hand['active']:
                continue
                
            palm_pos = hand['palm_center']
            
            # Special Ability: Whirlpool Vortex (wrist roll speed > 4.5 rad/s)
            rot_speed = hand['rotation_speed']
            if abs(rot_speed) > 4.5:
                # Spawn swirling water particles
                if np.sum(self.active) < self.max_particles:
                    count = 8
                    angles = np.random.uniform(0, 2*math.pi, count)
                    radii = np.random.uniform(20, 150, count)
                    px = palm_pos[0] + np.cos(angles) * radii
                    py = palm_pos[1] + np.sin(angles) * radii
                    vx = np.random.uniform(-30, 30, count)
                    vy = np.random.uniform(-30, 30, count)
                    colors = np.zeros((count, 3), dtype=np.uint8)
                    colors[:] = [0, 206, 209] # Turquoise
                    self.spawn_particles(count, px, py, vx, vy, colors, 
                                          np.random.uniform(2.5, 5.0, count), 
                                          np.random.uniform(1.0, 1.8, count), 
                                          np.random.uniform(0.4, 0.8, count), 
                                          attrib_type=0)
                
                # Apply vortex physics to nearby particles (within 300px)
                d = palm_pos - self.pos
                dist = np.linalg.norm(d, axis=1)
                near_mask = (dist < 300.0) & active_mask
                if np.any(near_mask):
                    d_near = d[near_mask]
                    dist_near = dist[near_mask][:, None]
                    u_rad = d_near / (dist_near + 1e-5)
                    u_tan = np.zeros_like(u_rad)
                    u_tan[:, 0] = -u_rad[:, 1]
                    u_tan[:, 1] = u_rad[:, 0]
                    
                    # Pull in and spin
                    pull_force = (dist_near - 20.0) * 5.0
                    self.vel[near_mask] = u_tan * rot_speed * 110.0 + u_rad * pull_force
                    self.color[near_mask] = [0, 206, 209] # Turquoise
                continue # Skip normal finger parting for this hand while spinning
            
            # Spawn liquid light particles when stirring joints normally
            if np.sum(self.active) < self.max_particles:
                tips = hand['landmarks'][[4, 8, 12, 16, 20], :2]
                for tip in tips:
                    if random.random() < 0.50:  # Increased from 0.35 for better density
                        px = tip[0] + np.random.uniform(-12, 12)
                        py = tip[1] + np.random.uniform(-12, 12)
                        # Initial velocity follows curl-noise direction at this point
                        freq = 0.010
                        t = self._time
                        vx = float(30.0 * math.cos(freq * py + t * 0.6))
                        vy = float(-30.0 * math.cos(freq * px + t * 0.55))
                        vx += np.random.uniform(-20, 20)
                        vy += np.random.uniform(-20, 20)
                        # Slight green-turquoise color variation
                        g_val = random.randint(220, 255)
                        b_val = random.randint(140, 210)
                        size = np.random.uniform(2.5, 5.5)
                        life = np.random.uniform(1.0, 2.2)
                        decay = np.random.uniform(0.25, 0.5)
                        self.spawn_particles(1, px, py, vx, vy, [0, g_val, b_val], size, life, decay, attrib_type=0)

            # Normal hydro parting force
            landmarks = hand['landmarks']
            joints = landmarks[:, :2]
            for j_idx in range(21):
                joint_pos = joints[j_idx]
                d = self.pos - joint_pos
                dist = np.linalg.norm(d, axis=1)
                
                part_mask = (dist < 45.0) & active_mask
                if np.any(part_mask):
                    d_part = d[part_mask]
                    dist_part = dist[part_mask][:, None]
                    u_dir = d_part / (dist_part + 1e-5)
                    push_force = 180.0 * (1.0 - (dist_part / 45.0))
                    self.vel[part_mask] += u_dir * push_force

    def _update_arc(self, gestures, active_mask, dt):
        # Arc: electric particles, slightly attracted to fingers
        # In main drawing loop we will draw lightning bolts, here we just do basic physics
        # Particles are attracted to fingers slightly
        for side in ['Left', 'Right']:
            hand = gestures[side]
            if not hand['active']:
                continue
            
            # Spawn electric spark particles at fingertips
            if np.sum(self.active) < self.max_particles:
                tips = hand['landmarks'][[4, 8, 12, 16, 20], :2]
                for tip in tips:
                    if random.random() < 0.22: # 22% chance per frame per tip
                        px = tip[0] + np.random.uniform(-10, 10)
                        py = tip[1] + np.random.uniform(-10, 10)
                        vx = np.random.uniform(-150, 150)
                        vy = np.random.uniform(-150, 150)
                        colors = [200, 230, 255] # Cold blue-white
                        size = np.random.uniform(1.5, 3.5)
                        life = np.random.uniform(0.3, 0.7)
                        decay = np.random.uniform(1.0, 2.0)
                        self.spawn_particles(1, px, py, vx, vy, colors, size, life, decay, attrib_type=0)

            # Fingertips attract particles slightly
            tips = hand['landmarks'][[4, 8, 12, 16, 20], :2]
            for tip in tips:
                d = tip - self.pos
                dist = np.linalg.norm(d, axis=1)
                
                attract_mask = (dist < 220.0) & active_mask
                if np.any(attract_mask):
                    d_att = d[attract_mask]
                    dist_att = dist[attract_mask][:, None]
                    u_dir = d_att / (dist_att + 1e-5)
                    
                    force = 120.0 * (1.0 - (dist_att / 220.0))
                    self.vel[attract_mask] += u_dir * force * dt
                    
                    # Glow brighter when attracted
                    self.size[attract_mask] = np.random.uniform(2.5, 4.0, np.sum(attract_mask))

    def get_lightning_arcs(self, gestures):
        """
        Calculates lightning segments to draw in ARC mode.
        Returns: list of dicts: {'p1': (x,y), 'p2': (x,y), 'intensity': float, 'forks': bool}
        """
        if self.mode != 5 or self.transition_timer > 0.0:
            return []
            
        arcs = []
        active_sides = [side for side in ['Left', 'Right'] if gestures[side]['active']]
        
        # 0. Storm Core Lightning (Arc Super)
        if self.super_active and self.super_type == 5:
            # Arcs to all active fingertips
            for side in active_sides:
                hand = gestures[side]
                landmarks = hand['landmarks']
                tips = [landmarks[4][:2], landmarks[8][:2], landmarks[12][:2], landmarks[16][:2], landmarks[20][:2]]
                for tip in tips:
                    arcs.append({
                        'p1': tuple(self.super_pos),
                        'p2': tuple(tip),
                        'intensity': np.random.uniform(0.8, 1.0),
                        'forks': True
                    })
            # Arcs to random borders of the screen
            for _ in range(3):
                border = random.choice(['top', 'bottom', 'left', 'right'])
                if border == 'top':
                    bx = random.uniform(0, self.width)
                    by = 0.0
                elif border == 'bottom':
                    bx = random.uniform(0, self.width)
                    by = self.height
                elif border == 'left':
                    bx = 0.0
                    by = random.uniform(0, self.height)
                else:
                    bx = self.width
                    by = random.uniform(0, self.height)
                    
                arcs.append({
                    'p1': tuple(self.super_pos),
                    'p2': (bx, by),
                    'intensity': np.random.uniform(0.7, 0.9),
                    'forks': True
                })
        
        # 1. Arc between hands if close
        if len(active_sides) == 2:
            l_wrist = gestures['Left']['wrist']
            r_wrist = gestures['Right']['wrist']
            dist = np.linalg.norm(l_wrist - r_wrist)
            
            if dist < 280.0:
                # Double-hand arc!
                arcs.append({
                    'p1': tuple(l_wrist),
                    'p2': tuple(r_wrist),
                    'intensity': 1.0 - (dist / 280.0),
                    'forks': True
                })

        # 2. Arc between fingers if spread wide
        for side in active_sides:
            hand = gestures[side]
            landmarks = hand['landmarks']
            
            # Check fingers spread: distance between index tip (8) and middle tip (12)
            # relative to middle MCP to PIP distance
            index_tip = landmarks[8][:2]
            middle_tip = landmarks[12][:2]
            ring_tip = landmarks[16][:2]
            pinky_tip = landmarks[20][:2]
            thumb_tip = landmarks[4][:2]
            
            hand_scale = max(1.0, np.linalg.norm(landmarks[9] - landmarks[0]))
            spread_dist = np.linalg.norm(index_tip - middle_tip) / hand_scale
            
            # If spread is wide (> 0.65 normalized scale)
            if spread_dist > 0.65:
                # Fork lightning between all 5 fingers across the hand!
                tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                for i in range(len(tips) - 1):
                    arcs.append({
                        'p1': tuple(tips[i]),
                        'p2': tuple(tips[i+1]),
                        'intensity': min(1.0, spread_dist * 0.7),
                        'forks': False
                    })
            
            # 3. Micro electric discharges from fingertips to nearest particles
            tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            active_particle_indices = np.where(self.active)[0]
            
            if len(active_particle_indices) > 0:
                particle_positions = self.pos[active_particle_indices]
                
                for tip in tips:
                    # Find distance to all active particles
                    dists = np.linalg.norm(particle_positions - tip, axis=1)
                    
                    # Connect to particles within 110px (max 2 particles per finger)
                    close_idx = np.where(dists < 110.0)[0]
                    if len(close_idx) > 0:
                        # Sort by distance
                        sorted_close = close_idx[np.argsort(dists[close_idx])][:2]
                        for idx in sorted_close:
                            part_pos = particle_positions[idx]
                            arcs.append({
                                'p1': tuple(tip),
                                'p2': tuple(part_pos),
                                'intensity': 0.3 + 0.7 * (1.0 - (dists[idx] / 110.0)),
                                'forks': False
                            })
                            
        return arcs

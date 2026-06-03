import pygame
import numpy as np
import math
import time

def draw_projected_circle(surface, center, normal, u_basis, v_basis, radius, color, width=1, dash_segments=None):
    """
    Draws a 3D circle projected onto a 2D Pygame surface.
    center: (3,) center of the circle in 3D
    normal: (3,) normal vector to the plane of the circle
    u_basis, v_basis: orthogonal basis vectors in the plane of the circle
    radius: radius of the circle
    color: (R, G, B, A) color tuple
    width: line thickness
    dash_segments: optional list of tuples (start_deg, end_deg) to draw segmented lines (like arc reactor)
    """
    points_3d = []
    steps = 64
    
    if dash_segments is None:
        dash_segments = [(0, 360)]
        
    for start_deg, end_deg in dash_segments:
        segment_pts = []
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        
        # Subdivide segment
        seg_steps = max(2, int(steps * (end_deg - start_deg) / 360.0))
        for i in range(seg_steps + 1):
            theta = start_rad + (end_rad - start_rad) * (i / seg_steps)
            pt = center + radius * math.cos(theta) * u_basis + radius * math.sin(theta) * v_basis
            segment_pts.append((int(pt[0]), int(pt[1])))
            
        if len(segment_pts) >= 2:
            pygame.draw.lines(surface, color, False, segment_pts, width)

_cached_fonts = {}

def get_font(size, bold=False):
    """
    Lazily retrieves/caches Consolas sysfont or fallback font to prevent loading overhead.
    """
    key = (size, bold)
    if key not in _cached_fonts:
        try:
            _cached_fonts[key] = pygame.font.SysFont("Consolas", size, bold=bold)
        except Exception:
            _cached_fonts[key] = pygame.font.Font(None, size)
    return _cached_fonts[key]

def draw_shadow_text(surface, text, font, pos, color, shadow_color=(0, 0, 0), offset=(1.5, 1.5)):
    """
    Renders text with a drop shadow for high legibility against dynamic backgrounds.
    """
    # Render shadow (cast alpha if the color has alpha)
    shadow_surf = font.render(text, True, shadow_color)
    if len(shadow_color) > 3:
        shadow_surf.set_alpha(shadow_color[3])
    surface.blit(shadow_surf, (pos[0] + offset[0], pos[1] + offset[1]))
    
    # Render main text
    text_surf = font.render(text, True, color)
    if len(color) > 3:
        text_surf.set_alpha(color[3])
    surface.blit(text_surf, pos)

def draw_shadow_text_centered(surface, text, font, center_pos, color, shadow_color=(0, 0, 0), offset=(1.5, 1.5)):
    """
    Renders text centered at center_pos with a drop shadow.
    """
    text_surf = font.render(text, True, color)
    if len(color) > 3:
        text_surf.set_alpha(color[3])
    text_rect = text_surf.get_rect(center=center_pos)
    
    shadow_surf = font.render(text, True, shadow_color)
    if len(shadow_color) > 3:
        shadow_surf.set_alpha(shadow_color[3])
    surface.blit(shadow_surf, (text_rect.x + offset[0], text_rect.y + offset[1]))
    surface.blit(text_surf, text_rect)

def draw_holographic_label(surface, text, center_pos, color):
    """
    Draws a futuristic holographic capsule box with text inside.
    """
    font = get_font(12, bold=True)
    text_surf = font.render(text, True, (255, 255, 255))
    text_w, text_h = text_surf.get_size()
    
    box_w = text_w + 24
    box_h = text_h + 10
    
    bx = center_pos[0] - box_w // 2
    by = center_pos[1] - box_h // 2
    
    # Render glass capsule background
    glass = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    glass.fill((0, 8, 16, 180)) # Semi-transparent dark blue-gray
    
    # Outer capsule border
    pygame.draw.rect(glass, color, (0, 0, box_w, box_h), 1)
    
    # Corner brackets (thicker lines at corners for tech look)
    tl = 5
    # Top Left
    pygame.draw.line(glass, (255, 255, 255), (0, 0), (tl, 0), 2)
    pygame.draw.line(glass, (255, 255, 255), (0, 0), (0, tl), 2)
    # Top Right
    pygame.draw.line(glass, (255, 255, 255), (box_w, 0), (box_w - tl, 0), 2)
    pygame.draw.line(glass, (255, 255, 255), (box_w, 0), (box_w, tl), 2)
    # Bottom Left
    pygame.draw.line(glass, (255, 255, 255), (0, box_h), (tl, box_h), 2)
    pygame.draw.line(glass, (255, 255, 255), (0, box_h), (0, box_h - tl), 2)
    # Bottom Right
    pygame.draw.line(glass, (255, 255, 255), (box_w, box_h), (box_w - tl, box_h), 2)
    pygame.draw.line(glass, (255, 255, 255), (box_w, box_h), (box_w, box_h - tl), 2)
    
    # Blit glass
    surface.blit(glass, (bx, by))
    
    # Blit text centered with drop shadow
    draw_shadow_text_centered(surface, text, font, center_pos, (255, 255, 255), shadow_color=(0, 0, 0))

MODE_COLORS = {
    1: (0, 212, 255),    # Surge (electric blue)
    2: (255, 100, 0),    # Inferno (orange)
    3: (138, 43, 226),   # Void (purple)
    4: (0, 250, 154),    # Current (spring green)
    5: (200, 230, 255)   # Arc (cold blue/white)
}

def get_active_ability_text(hand, mode, super_active):
    """
    Determines the current technical text state for a hand based on gestures and physics mode.
    """
    avg_curl = float(np.mean(hand['curls'][1:]))
    pinch_active = hand['pinch_active']
    
    if super_active:
        if mode == 1:
            return "SUPER: PLASMA BLAST"
        elif mode == 2:
            return "SUPER: PYROCLASTIC SUN"
        elif mode == 3:
            return "SUPER: COSMIC CLAP"
        elif mode == 4:
            return "SUPER: TSUNAMI SWEEP"
        elif mode == 5:
            return "SUPER: STORM CORE"
            
    if mode == 1:  # SURGE
        if pinch_active['index']:
            return "SPECIAL: RAILGUN READY"
        elif avg_curl > 0.75:
            return "SURGE: ATTRACT WELL"
        elif avg_curl < 0.35:
            return "SURGE: REPEL EMISSION"
        else:
            return "SURGE: ENERGY STANDBY"
            
    elif mode == 2:  # INFERNO
        index_curl = float(hand['curls'][1])
        mid_curl   = float(hand['curls'][2])
        ring_curl  = float(hand['curls'][3])
        gun_shape  = (index_curl < 0.33 and mid_curl > 0.56 and ring_curl > 0.56)
        if gun_shape:
            return "SPECIAL: FIREBALL CHARGING"
        elif avg_curl > 0.8:
            return "INFERNO: SMOTHER HEAT"
        elif index_curl < 0.35:
            return "INFERNO: FIRE PILLAR"
        else:
            return "INFERNO: EMBER STANDBY"
            
    elif mode == 3:  # VOID
        v_idx  = float(hand['curls'][1])
        v_mid  = float(hand['curls'][2])
        v_ring = float(hand['curls'][3])
        v_pky  = float(hand['curls'][4])
        v_sign = (v_idx < 0.36 and v_mid < 0.41 and v_ring > 0.54 and v_pky > 0.54)
        if v_sign:
            return "SPECIAL: VOID RIFT CHARGE"
        elif avg_curl > 0.75:
            return "VOID: GRAVITY CLENCH"
        else:
            return "VOID: QUANTUM STANDBY"
            
    elif mode == 4:  # CURRENT
        if abs(hand['rotation_speed']) > 4.5:
            return "SPECIAL: VORTEX SPIN"
        else:
            return "CURRENT: LIQUID CARVING"
            
    elif mode == 5:  # ARC
        # Spread calculation
        landmarks = hand['landmarks']
        hand_scale = max(1.0, np.linalg.norm(landmarks[9] - landmarks[0]))
        index_tip = landmarks[8][:2]
        middle_tip = landmarks[12][:2]
        spread_dist = np.linalg.norm(index_tip - middle_tip) / hand_scale
        
        if spread_dist > 0.65:
            return "SPECIAL: CHAIN ARC SPREAD"
        else:
            return "ARC: SYSTEM IONIZING"
            
    return "HEADING CALIBRATED"

def draw_hand_overlay(surface, gestures, time_val, physics=None):
    """
    Draws the technical augmented anatomy overlay for all active hands.
    gestures: dict of gestures containing landmarks, curls, pinch info, etc.
    time_val: float value representing current time (for pulsing effects)
    physics: optional ParticleSystem instance to determine dynamic styling/modes
    """
    mode = physics.mode if physics is not None else 1
    super_active = physics.super_active if physics is not None else False
    overlay_color = MODE_COLORS.get(mode, (0, 212, 255))
    white = (255, 255, 255)
    
    # Bone connections definition (21 landmarks)
    finger_connections = [
        # Thumb
        [0, 1, 2, 3, 4],
        # Index
        [0, 5, 6, 7, 8],
        # Middle
        [0, 9, 10, 11, 12],
        # Ring
        [0, 13, 14, 15, 16],
        # Pinky
        [0, 17, 18, 19, 20],
        # Palm connections
        [5, 9, 13, 17]
    ]

    for side in ['Left', 'Right']:
        hand = gestures[side]
        if not hand['active']:
            continue
            
        landmarks = hand['landmarks']  # (21, 3)
        curls = hand['curls']          # (5,)
        pinch_active = hand['pinch_active']
        
        # 1. Calculate general hand parameters
        wrist_pos = landmarks[0]
        middle_mcp = landmarks[9]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        # Construct 3D Palm Basis for projection
        v_up = middle_mcp - wrist_pos
        v_width = pinky_mcp - index_mcp
        
        # Normal vector to the palm
        normal = np.cross(v_up, v_width)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-4:
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            normal /= norm_val
            
        # Plane basis vectors
        u_basis = v_width / (np.linalg.norm(v_width) + 1e-5)
        v_basis = np.cross(normal, u_basis)
        v_basis /= (np.linalg.norm(v_basis) + 1e-5)

        # Pulse multiplier
        pulse = 1.0 + 0.18 * math.sin(time_val * 11.0)
        
        # 1.5. Estimate and Draw Cybernetic Forearm
        v_forearm = wrist_pos - middle_mcp
        v_forearm_norm = np.linalg.norm(v_forearm)
        if v_forearm_norm > 1e-3:
            v_forearm_dir = v_forearm / v_forearm_norm
        else:
            v_forearm_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
        hand_len = max(1.0, np.linalg.norm(middle_mcp - wrist_pos))
        arm_len = hand_len * 2.3
        elbow_pos = wrist_pos + v_forearm_dir * arm_len
        
        # Radius & Ulna bones (offsets from central forearm vector)
        wrist_width_vec = u_basis * 12.0
        elbow_width_vec = u_basis * 20.0
        
        p_wrist_radius = wrist_pos + wrist_width_vec
        p_wrist_ulna = wrist_pos - wrist_width_vec
        p_elbow_radius = elbow_pos + elbow_width_vec
        p_elbow_ulna = elbow_pos - elbow_width_vec
        
        p_wr_int = (int(p_wrist_radius[0]), int(p_wrist_radius[1]))
        p_wu_int = (int(p_wrist_ulna[0]), int(p_wrist_ulna[1]))
        p_er_int = (int(p_elbow_radius[0]), int(p_elbow_radius[1]))
        p_eu_int = (int(p_elbow_ulna[0]), int(p_elbow_ulna[1]))
        
        pygame.draw.line(surface, (*overlay_color, 110), p_wr_int, p_er_int, 2)
        pygame.draw.line(surface, (*overlay_color, 110), p_wu_int, p_eu_int, 2)
        
        # Central energy conduit with moving pulses
        p_wrist_core = (int(wrist_pos[0]), int(wrist_pos[1]))
        p_elbow_core = (int(elbow_pos[0]), int(elbow_pos[1]))
        pygame.draw.line(surface, (255, 255, 255, 90), p_wrist_core, p_elbow_core, 1)
        
        num_nodes = 3
        for n_idx in range(num_nodes):
            node_t = ((time_val * 0.7) + (n_idx / num_nodes)) % 1.0
            node_pos = wrist_pos + (elbow_pos - wrist_pos) * node_t
            nx, ny = int(node_pos[0]), int(node_pos[1])
            ns = int(3 + 1.2 * math.sin(time_val * 14.0 + n_idx))
            pygame.draw.circle(surface, white, (nx, ny), ns)
            pygame.draw.circle(surface, (*overlay_color, 150), (nx, ny), ns + 2, 1)
            
        # Holographic transverse rings/cross-sections
        for r_idx in [0.25, 0.5, 0.75]:
            ring_center = wrist_pos + (elbow_pos - wrist_pos) * r_idx
            ring_r = 13.0 + (20.0 - 13.0) * r_idx
            
            r_normal = v_forearm_dir
            r_u = u_basis
            r_v = np.cross(r_normal, r_u)
            r_v_norm = np.linalg.norm(r_v)
            if r_v_norm > 1e-3:
                r_v /= r_v_norm
            else:
                r_v = v_basis
                
            draw_projected_circle(
                surface, ring_center, r_normal, r_u, r_v,
                radius=ring_r,
                color=(*overlay_color, 80),
                width=1,
                dash_segments=[(0, 100), (120, 220), (240, 340)]
            )
            
        # 2. Draw Tendon Lines (Wrist to base of fingers, visible when partially curled)
        mcp_indices = [5, 9, 13, 17]
        for idx, mcp_idx in enumerate(mcp_indices):
            finger_curl = curls[idx + 1]  # index 1-4
            tendon_intensity = math.sin(finger_curl * math.pi)
            if tendon_intensity > 0.05:
                tendon_alpha = int(tendon_intensity * 130)
                color = (*overlay_color, tendon_alpha)
                
                p1 = (int(wrist_pos[0]), int(wrist_pos[1]))
                p2 = (int(landmarks[mcp_idx][0]), int(landmarks[mcp_idx][1]))
                pygame.draw.line(surface, color, p1, p2, 1)

        # 3. Draw Bone Segments (technical skeleton)
        for connection in finger_connections:
            for i in range(len(connection) - 1):
                p1_idx = connection[i]
                p2_idx = connection[i+1]
                
                p1 = (int(landmarks[p1_idx][0]), int(landmarks[p1_idx][1]))
                p2 = (int(landmarks[p2_idx][0]), int(landmarks[p2_idx][1]))
                
                color = (*overlay_color, 140)
                pygame.draw.line(surface, color, p1, p2, 1)

        # 4. Draw Knuckle Arcs (technical plating over joints)
        knuckles = [5, 9, 13, 17, 6, 10, 14, 18]
        for kn in knuckles:
            kn_pos = landmarks[kn]
            
            next_lm = landmarks[kn + 1]
            dir_vec = next_lm - kn_pos
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > 1e-3:
                dir_vec /= dir_norm
            else:
                dir_vec = v_up / (np.linalg.norm(v_up) + 1e-5)
                
            perp_vec = np.cross(normal, dir_vec)
            perp_vec /= (np.linalg.norm(perp_vec) + 1e-5)
            
            draw_projected_circle(
                surface, kn_pos, normal, dir_vec, perp_vec,
                radius=11.0,
                color=(*overlay_color, 90),
                width=1,
                dash_segments=[(-60, 60)]
            )

        # 5. Draw Wrist Core (3D holographic arc reactor)
        rot_offset = (time_val * 45.0) % 360.0
        
        draw_projected_circle(
            surface, wrist_pos, normal, u_basis, v_basis,
            radius=34.0,
            color=(*overlay_color, int(150 * pulse)),
            width=2,
            dash_segments=[
                (rot_offset, rot_offset + 50),
                (rot_offset + 90, rot_offset + 140),
                (rot_offset + 180, rot_offset + 230),
                (rot_offset + 270, rot_offset + 320)
            ]
        )
        
        draw_projected_circle(
            surface, wrist_pos, normal, u_basis, v_basis,
            radius=20.0,
            color=(*overlay_color, 90),
            width=1
        )
        
        center_color = white if pulse > 1.05 else overlay_color
        pygame.draw.circle(surface, center_color, (int(wrist_pos[0]), int(wrist_pos[1])), 4)
        pygame.draw.circle(surface, (*overlay_color, 80), (int(wrist_pos[0]), int(wrist_pos[1])), 10)

        # 6. Draw Joint Nodes and Fingertip Coronas
        for i in range(21):
            pos = landmarks[i]
            x, y = int(pos[0]), int(pos[1])
            is_tip = i in [4, 8, 12, 16, 20]
            
            if is_tip:
                tip_names = {4: 'thumb', 8: 'index', 12: 'middle', 16: 'ring', 20: 'pinky'}
                finger_name = tip_names[i]
                
                is_pinching = False
                if finger_name == 'thumb':
                    is_pinching = any(pinch_active.values())
                else:
                    is_pinching = pinch_active[finger_name]
                
                if is_pinching:
                    r_corona = int(14.0 * pulse)
                    pygame.draw.circle(surface, (*overlay_color, 45), (x, y), r_corona + 10)
                    pygame.draw.circle(surface, (*overlay_color, 80), (x, y), r_corona + 4)
                    pygame.draw.circle(surface, white, (x, y), 6)
                else:
                    r_corona = int(8.0 * pulse)
                    pygame.draw.circle(surface, (*overlay_color, 60), (x, y), r_corona + 5)
                    pygame.draw.circle(surface, (*overlay_color, 120), (x, y), r_corona)
                    pygame.draw.circle(surface, white, (x, y), 3.5)
            else:
                pygame.draw.circle(surface, white, (x, y), 2)
                pygame.draw.circle(surface, (*overlay_color, 150), (x, y), 4, 1)

        # 7. Render Floating Holographic Label
        ability_text = get_active_ability_text(hand, mode, super_active)
        
        # Position label centered above middle MCP joint (landmark 9)
        palm_pos = hand['palm_center']
        label_x = int(np.clip(palm_pos[0], 120, surface.get_width() - 120))
        label_y = int(np.clip(palm_pos[1] - 110, 40, surface.get_height() - 40))
        
        label_pos = (label_x, label_y)
        
        # Connection line to palm center
        line_start = (label_pos[0], label_pos[1] + 10)
        line_end = (int(palm_pos[0]), int(palm_pos[1]))
        
        # Technical dashed line
        pygame.draw.line(surface, (*overlay_color, 120), line_start, line_end, 1)
        pygame.draw.circle(surface, (255, 255, 255), line_start, 2)
        pygame.draw.circle(surface, overlay_color, line_end, 3)
        
        # Draw holographic text box capsule
        draw_holographic_label(surface, ability_text, label_pos, overlay_color)

    # ─── Gojo Infinity Veil (Void Mode) ─────────────────────────────────────────
    if mode == 3 and gestures.get('gojo_infinity', False):
        left  = gestures['Left']
        right = gestures['Right']
        if left['active'] and right['active']:
            mid_x = int((left['palm_center'][0] + right['palm_center'][0]) * 0.5)
            mid_y = int((left['palm_center'][1] + right['palm_center'][1]) * 0.5)
            
            VOID_PURPLE   = (138, 43, 226)
            VOID_LAVENDER = (180, 120, 255)
            
            # ── Draw X-cross at each hand's middle+ring fingertips ──
            for hand_ref in (left, right):
                lms = hand_ref['landmarks']
                tip_mid  = (int(lms[12][0]), int(lms[12][1]))
                tip_ring = (int(lms[16][0]), int(lms[16][1]))
                cx = (tip_mid[0] + tip_ring[0]) // 2
                cy = (tip_mid[1] + tip_ring[1]) // 2
                r  = 14
                cross_alpha = int(180 + 60 * math.sin(time_val * 9.0))
                pygame.draw.line(surface, (*VOID_LAVENDER, cross_alpha),
                                 (cx - r, cy - r), (cx + r, cy + r), 3)
                pygame.draw.line(surface, (*VOID_LAVENDER, cross_alpha),
                                 (cx + r, cy - r), (cx - r, cy + r), 3)
                # Small glow circle at cross center
                glow_s = pygame.Surface((r * 3, r * 3), pygame.SRCALPHA)
                glow_a = int(60 + 40 * math.sin(time_val * 7.0))
                pygame.draw.circle(glow_s, (*VOID_PURPLE, glow_a), (r + r // 2, r + r // 2), r)
                surface.blit(glow_s, (cx - r - r // 2, cy - r - r // 2))
            
            # ── Hexagonal rings at midpoint ──
            for ring_idx in range(4):
                phase   = time_val * 1.8 + ring_idx * (math.pi / 2.0)
                base_r  = 55 + ring_idx * 38
                pulse_r = base_r + 8.0 * math.sin(phase)
                alpha   = max(30, int(210 - ring_idx * 45 + 40 * math.sin(phase + 1.0)))
                hex_pts = []
                for k in range(7):
                    angle = math.pi / 6 + k * (math.pi / 3.0) + time_val * 0.35
                    hx = mid_x + pulse_r * math.cos(angle)
                    hy = mid_y + pulse_r * math.sin(angle)
                    hex_pts.append((int(hx), int(hy)))
                hex_color = (*VOID_LAVENDER, alpha) if ring_idx % 2 == 0 else (*VOID_PURPLE, alpha)
                try:
                    pygame.draw.lines(surface, hex_color, True, hex_pts, max(1, 3 - ring_idx))
                except Exception:
                    pass
                if ring_idx == 0:
                    gs = pygame.Surface((20, 20), pygame.SRCALPHA)
                    pygame.draw.circle(gs, (*VOID_LAVENDER, int(80 + 60 * math.sin(time_val * 9.0))), (10, 10), 10)
                    surface.blit(gs, (mid_x - 10, mid_y - 10))
            
            # ── Radial spokes ──
            for k in range(6):
                angle     = k * (math.pi / 3.0) + time_val * 0.6
                spoke_len = 55.0 + 20.0 * math.sin(time_val * 2.5 + k)
                sx = mid_x + spoke_len * math.cos(angle)
                sy = mid_y + spoke_len * math.sin(angle)
                pygame.draw.line(surface, (*VOID_LAVENDER, int(140 + 80 * math.sin(time_val * 8.0 + k))),
                                 (mid_x, mid_y), (int(sx), int(sy)), 2)
            
            # ── ∞ INFINITY label ──
            inf_font   = get_font(22, bold=True)
            label_surf = inf_font.render("∞ INFINITY", True, VOID_LAVENDER)
            lw, lh     = label_surf.get_size()
            label_y    = int(mid_y - 90 + 5 * math.sin(time_val * 5.0))
            pill = pygame.Surface((lw + 24, lh + 10), pygame.SRCALPHA)
            pill.fill((10, 0, 25, 180))
            pygame.draw.rect(pill, (*VOID_LAVENDER, 180), (0, 0, lw + 24, lh + 10), 1)
            surface.blit(pill,       (mid_x - (lw + 24) // 2, label_y - 5))
            surface.blit(label_surf, (mid_x - lw // 2, label_y))

    # ─── Torch Raise visual (Inferno Super arm) ──────────────────────────────────
    if mode == 2 and gestures.get('torch_raise', False):
        TORCH_ORANGE = (255, 160, 30)
        TORCH_WHITE  = (255, 245, 200)
        for hand_ref in (gestures['Left'], gestures['Right']):
            if not hand_ref['active']:
                continue
            # Draw a blazing aura around index+middle tips
            for tip_idx in (8, 12):
                lms = hand_ref['landmarks']
                tx, ty = int(lms[tip_idx][0]), int(lms[tip_idx][1])
                for r in (18, 10):
                    pulse_a = int(90 + 70 * math.sin(time_val * 12.0 + tip_idx))
                    gs = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(gs, (*TORCH_ORANGE, pulse_a), (r, r), r)
                    surface.blit(gs, (tx - r, ty - r))
                # Small bright core
                pygame.draw.circle(surface, TORCH_WHITE, (tx, ty), 4)

def draw_electricity_arcs(surface, arcs, time_val):
    """
    Renders electric lightning bolts onto the surface.
    arcs: list of arcs returned from the physics system
    """
    for arc in arcs:
        p1 = arc['p1']
        p2 = arc['p2']
        intensity = arc['intensity']
        forks = arc['forks']
        
        # Set thickness and color based on intensity
        thickness = max(1, int(3.5 * intensity))
        color_core = (255, 255, 255, int(255 * intensity))
        color_glow = (0, 212, 255, int(130 * intensity))
        
        # Subdivide and generate jagged points
        pts = generate_lightning_path(p1, p2, segments=12, displacement=15.0)
        
        # Draw outer glow
        if len(pts) >= 2:
            try:
                pygame.draw.lines(surface, color_glow, False, pts, thickness + 4)
                pygame.draw.lines(surface, color_core, False, pts, thickness)
            except Exception as e:
                print(f"[HEXIS Debug] main lines failed. pts={pts}, type={type(pts)}, pt_types={[type(x) for x in pts]}, Error: {e}")
                raise e
            
        # Draw small forks if requested
        if forks and len(pts) >= 4:
            # Fork 1: branch from middle point of lightning
            mid_idx = len(pts) // 2
            branch_start = pts[mid_idx]
            
            # Find vector direction
            v_dir = np.array(p2) - np.array(p1)
            v_norm = np.linalg.norm(v_dir)
            if v_norm > 1e-3:
                v_dir = v_dir / v_norm
                # Perpendicular vector
                v_perp = np.array([-v_dir[1], v_dir[0]])
                
                # Branch destination: offset forward and outward
                branch_end = branch_start + v_dir * (v_norm * 0.3) + v_perp * np.random.uniform(-40, 40)
                branch_pts = generate_lightning_path(tuple(branch_start), tuple(branch_end), segments=6, displacement=8.0)
                
                try:
                    pygame.draw.lines(surface, color_glow, False, branch_pts, max(1, thickness - 1) + 2)
                    pygame.draw.lines(surface, color_core, False, branch_pts, max(1, thickness - 2))
                except Exception as e:
                    print(f"[HEXIS Debug] fork lines failed. branch_pts={branch_pts}, Error: {e}")
                    raise e

def generate_lightning_path(p1, p2, segments=10, displacement=12.0):
    """
    Generates a jagged line path from p1 to p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    path = [p1]
    
    # Vector from p1 to p2
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 5.0:
        return [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))]
        
    v_norm = v / length
    # Perpendicular unit vector
    perp = np.array([-v_norm[1], v_norm[0]])
    
    for i in range(1, segments):
        t = i / segments
        # Interpolate along line
        base_pos = p1 + t * v
        # Add random perpendicular offset
        # Offset scales down towards the endpoints to ensure they meet p1 and p2 exactly
        scale = math.sin(t * math.pi)  # 0 at start, 1 at mid, 0 at end
        offset = np.random.uniform(-displacement, displacement) * scale
        path.append(base_pos + perp * offset)
        
    path.append(p2)
    return [(int(pt[0]), int(pt[1])) for pt in path]

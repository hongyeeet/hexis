import pygame
import cv2
import numpy as np
import time
import math
import sys

from camera import CameraTracker
from gestures import GestureAnalyzer
from physics import ParticleSystem
from overlay import draw_hand_overlay, draw_electricity_arcs, draw_shadow_text
import effects

# Configuration
WIDTH = 960
HEIGHT = 540
FPS = 60

# Mode Metadata
MODE_INFO = {
    1: {"name": "SURGE",   "icon": "⚡",  "desc": "Surge: Normal: Open palm repel/fist attract | Spec: Index point+flick railgun | Super: Flat-palm flick blast", "color": (0, 212, 255)},
    2: {"name": "INFERNO", "icon": "🔥",  "desc": "Inferno: Normal: ☞ point = fire pillar | Spec: ☞ gun shape+flick fireball | Super: Both ☞☞ raise (peace sign)", "color": (255, 100, 0)},
    3: {"name": "VOID",    "icon": "🌑",  "desc": "Void: Normal: Fist gravity well | Spec: ✌ V-sign+flick rift | Super: Cross 🤞🤞 middle+ring (Gojo ∞)", "color": (138, 43, 226)},
    4: {"name": "CURRENT", "icon": "🌊",  "desc": "Current: Normal: Stir to flow | Spec: Wrist spin whirlpool | Super: Parallel sweep tsunami", "color": (0, 250, 154)},
    5: {"name": "ARC",     "icon": "🎩️", "desc": "Arc: Normal: Fingertip sparks | Spec: Spread fingers chain | Super: Wrists-close storm core", "color": (200, 230, 255)},
}

def generate_virtual_landmarks(mx, my, pinch_pressed, curl_pressed, rotation_angle):
    """
    Generates a full 3D MediaPipe-like 21 landmark set for a virtual hand centered at (mx, my).
    Allows testing gestures without a camera.
    """
    lms = np.zeros((21, 3), dtype=np.float32)
    
    # 1. Base open hand layout relative to wrist
    base_offsets = {
        0:  [0, 80],     # Wrist
        # Thumb
        1:  [-25, 45],   # CMC
        2:  [-45, 25],   # MCP
        3:  [-55, 10],   # IP
        4:  [-62, -2],   # TIP
        # Index
        5:  [-20, 10],   # MCP
        6:  [-25, -20],  # PIP
        7:  [-28, -45],  # DIP
        8:  [-30, -65],  # TIP
        # Middle
        9:  [0, 5],      # MCP
        10: [0, -30],    # PIP
        11: [0, -60],    # DIP
        12: [0, -82],    # TIP
        # Ring
        13: [20, 10],    # MCP
        14: [22, -20],   # PIP
        15: [24, -45],   # DIP
        16: [25, -65],   # TIP
        # Pinky
        17: [38, 20],    # MCP
        18: [45, -5],    # PIP
        19: [48, -25],   # DIP
        20: [50, -42],   # TIP
    }
    
    for i, offset in base_offsets.items():
        lms[i, 0] = offset[0]
        lms[i, 1] = offset[1]
        lms[i, 2] = 0.0 # Z starts flat
        
    # Rotate around palm center (0, 0)
    cos_a = math.cos(rotation_angle)
    sin_a = math.sin(rotation_angle)
    
    for i in range(21):
        x, y = lms[i, 0], lms[i, 1]
        lms[i, 0] = x * cos_a - y * sin_a + mx
        lms[i, 1] = x * sin_a + y * cos_a + my
        # Add slight dummy Z depth mapping
        lms[i, 2] = -10.0 + (i % 5) * 5.0

    # 2. Apply Pinch (touch index 8 and thumb 4 tips)
    if pinch_pressed:
        mid_x = (lms[4, 0] + lms[8, 0]) * 0.5
        mid_y = (lms[4, 1] + lms[8, 1]) * 0.5
        lms[4, 0] = lms[8, 0] = mid_x
        lms[4, 1] = lms[8, 1] = mid_y
        lms[4, 2] = lms[8, 2] = 0.0

    # 3. Apply Curl (bend fingers index, middle, ring, pinky inwards)
    if curl_pressed:
        # Move PIP, DIP, TIP closer to MCP for all 4 fingers
        fingers = {
            'index':  (5, [6, 7, 8]),
            'middle': (9, [10, 11, 12]),
            'ring':   (13, [14, 15, 16]),
            'pinky':  (17, [18, 19, 20])
        }
        for f_name, (mcp_idx, sub_indices) in fingers.items():
            mcp = lms[mcp_idx]
            for idx in sub_indices:
                # Interpolate 80% towards MCP
                lms[idx] = mcp + (lms[idx] - mcp) * 0.2
                
    return lms
 
def draw_hud(surface, physics, fps, hands_active, gestures, virtual_active, show_help):
    """
    Draws a beautiful technical HUD overlay displaying metrics and mode information.
    """
    hud_font = pygame.font.SysFont("Consolas", 14)
    title_font = pygame.font.SysFont("Consolas", 24, bold=True)
    icon_font = pygame.font.SysFont("Segoe UI Symbol", 28)
    
    mode = physics.mode
    color = MODE_INFO[mode]["color"]
    
    # 1. Mode Banner Top-Left
    mode_text = MODE_INFO[mode]["name"]
    mode_desc = MODE_INFO[mode]["desc"]
    mode_icon = MODE_INFO[mode]["icon"]
    
    # Background glass box
    glass = pygame.Surface((440, 115), pygame.SRCALPHA)
    glass.fill((0, 10, 20, 160))
    pygame.draw.rect(glass, (*color, 120), (0, 0, 440, 115), 1)
    surface.blit(glass, (20, 20))
    
    # Render Icon and Text (use dropshadow text for sharpness)
    icon_surf = icon_font.render(mode_icon, True, color)
    surface.blit(icon_surf, (35, 30))
    draw_shadow_text(surface, mode_text, title_font, (85, 30), (255, 255, 255))
    
    # Split description by pipe to fit within box bounds
    desc_parts = [p.strip() for p in mode_desc.split('|')]
    for idx, part in enumerate(desc_parts):
        draw_shadow_text(surface, part, hud_font, (35, 75 + idx * 16), (180, 200, 220))
    
    # 2. System Status Top-Right
    status_glass = pygame.Surface((280, 135), pygame.SRCALPHA)
    status_glass.fill((0, 10, 20, 160))
    pygame.draw.rect(status_glass, (*color, 120), (0, 0, 280, 135), 1)
    surface.blit(status_glass, (WIDTH - 300, 20))
    
    fps_color = (0, 255, 128) if fps > 50 else (255, 100, 0)
    camera_status = "VIRTUAL EMULATION" if virtual_active else f"WEBCAM INDEX {getattr(physics, 'webcam_index', 0)}"
    cam_color = color if not virtual_active else (255, 200, 0)
    
    draw_shadow_text(surface, f"SYSTEM FPS  : {fps:.1f}", hud_font, (WIDTH - 280, 35), fps_color)
    draw_shadow_text(surface, f"INPUT SOURCE: {camera_status}", hud_font, (WIDTH - 280, 55), cam_color)
    draw_shadow_text(surface, f"HANDS ACTIVE: {hands_active}", hud_font, (WIDTH - 280, 75), (255, 255, 255))
    
    # Gesture metrics per hand
    y_offset = 95
    for side in ['Left', 'Right']:
        hand = gestures[side]
        if hand['active']:
            curl_avg = int(np.mean(hand['curls'][1:]) * 100)
            pinch_idx = "ON" if hand['pinch_active']['index'] else "OFF"
            metrics_text = f"{side.upper()} HAND : CURL {curl_avg}% | PINCH {pinch_idx}"
            draw_shadow_text(surface, metrics_text, hud_font, (WIDTH - 280, y_offset), color)
            y_offset += 20
            
    # 3. Help Instructions Bottom-Left
    if show_help:
        help_glass = pygame.Surface((920, 205), pygame.SRCALPHA)
        help_glass.fill((0, 5, 15, 210))
        pygame.draw.rect(help_glass, (*color, 180), (0, 0, 920, 205), 1)
        surface.blit(help_glass, (20, HEIGHT - 225))
        
        lines = [
            "HEXIS — AUGMENTED ANATOMY SUPERPOWERS HUD",
            "------------------------------------------------------------------------",
            " [SPACEBAR] : Manual Cycle Element   | [V]       : Toggle Emulation",
            " [C]        : Cycle Camera Index     | [F]/[F11] : Toggle Fullscreen",
            " [R]        : Reset Particles        | [H]       : Toggle Help Overlay",
            " ⚡ SURGE   : Normal: Open palm/fist    | Special: ☞ point+flick     | Super: Flat-palm FLICK",
            " 🔥 INFERNO : Normal: ☞ index up = pillar  | Special: ☞ gun+flick ball | Super: Both ☞☞ up (peace)",
            " 🌑 VOID    : Normal: Fist gravity well   | Special: ✌ V-sign+flick   | Super: Cross 🤞🤞 (Gojo ∞)",
            " 🌊 CURRENT : Normal: Stir joints         | Special: Wrist Spin      | Super: Parallel sweep",
            " 🎩️ ARC     : Normal: Move hand           | Special: Spread fingers  | Super: Wrists close",
        ]
        
        for idx, line in enumerate(lines):
            c = color if idx in [0, 1] else (255, 255, 255)
            draw_shadow_text(surface, line, hud_font, (35, HEIGHT - 210 + idx * 18), c)
    else:
        draw_shadow_text(surface, "Press [H] for Help / Controls Overlay", hud_font, (20, HEIGHT - 35), (*color, 180))
        
    # 4. Super Ability Activated Alert Banner
    if physics.super_active:
        super_glow = int(130 + 125 * math.sin(time.time() * 22.0))
        alert_font = pygame.font.SysFont("Consolas", 18, bold=True)
        temp_surf = alert_font.render(f"◆◇◆ SUPER ABILITY ACTIVATED ◆◇◆", True, (255, 255, 255))
        w = temp_surf.get_width()
        
        # Center horizontally
        ax = (WIDTH - w) // 2
        ay = HEIGHT - 265 if show_help else HEIGHT - 75
        
        # Flashing glass box
        alert_box = pygame.Surface((w + 40, 36), pygame.SRCALPHA)
        alert_box.fill((0, 10, 20, 210))
        pygame.draw.rect(alert_box, (*color, super_glow), (0, 0, w + 40, 36), 2)
        
        surface.blit(alert_box, (ax - 20, ay - 8))
        draw_shadow_text(surface, f"◆◇◆ SUPER ABILITY ACTIVATED ◆◇◆", alert_font, (ax, ay), (255, 255, 255))

def main():
    pygame.init()
    pygame.mixer.quit() # Disable mixer to save memory/prevent threads issues
    
    # Setup window
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.SCALED)
    pygame.display.set_caption("HEXIS — Augmented Anatomy Particle System")
    
    clock = pygame.time.Clock()
    
    # Initialize trackers and systems
    webcam_index = 0
    tracker = CameraTracker(width=WIDTH, height=HEIGHT, webcam_index=webcam_index)
    tracker.start()
    
    analyzer = GestureAnalyzer()
    physics = ParticleSystem(width=WIDTH, height=HEIGHT)
    physics.webcam_index = webcam_index
    warper = effects.SpaceWarper(size=260, intensity=0.45, sigma=65.0)
    
    # App State
    virtual_hand_active = not tracker.camera_active
    show_help = True
    virtual_rotation = 0.0
    shake_timer = 0.0
    shake_intensity = 0.0
    
    running = True
    time_start = time.time()
    
    # Separate surface for drawing elements to apply bloom on
    foreground_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        # Prevent huge dt spikes during startup
        if dt > 0.15:
            dt = 1.0/60.0
            
        time_val = time.time() - time_start
        
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Manual Mode Cycle
                    next_m = (physics.mode % 5) + 1
                    physics.trigger_mode_switch(next_m)
                    print(f"[HEXIS Mode] Switching to mode {next_m}")
                elif event.key == pygame.K_v:
                    virtual_hand_active = not virtual_hand_active
                    print(f"[HEXIS Emulation] Toggle virtual hand: {virtual_hand_active}")
                elif event.key == pygame.K_f or event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                    print("[HEXIS Display] Toggled Fullscreen Mode.")
                elif event.key == pygame.K_c:
                    webcam_index = (webcam_index + 1) % 4
                    physics.webcam_index = webcam_index
                    print(f"[HEXIS Camera] Cycling camera to index {webcam_index}...")
                    tracker.stop()
                    tracker = CameraTracker(width=WIDTH, height=HEIGHT, webcam_index=webcam_index)
                    tracker.start()
                elif event.key == pygame.K_r:
                    physics.reset_mode_particles(physics.mode)
                    print("[HEXIS Physics] Reset particles.")
                elif event.key == pygame.K_h:
                    show_help = not show_help
            
            # Virtual rotation control via mouse scroll wheel
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if virtual_hand_active:
                    if event.button == 4: # Scroll Up
                        virtual_rotation += 0.15
                    elif event.button == 5: # Scroll Down
                        virtual_rotation -= 0.15

        # 2. Retrieve tracking data
        bg_frame, raw_hands = tracker.get_data()
        
        # Fallback to dark background if camera is not active or failing
        if bg_frame is None or not tracker.camera_active:
            bg_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
        # Process background (darken and desaturate BGR image)
        processed_bg = effects.process_background(bg_frame)
        
        # Set hand tracking data source (Real vs Virtual)
        gestures_data = {}
        hands_active_count = 0
        
        if virtual_hand_active:
            # Emulated Hand at Mouse Position
            mx, my = pygame.mouse.get_pos()
            mouse_btns = pygame.mouse.get_pressed()
            
            pinch_pressed = mouse_btns[0] # Left click
            curl_pressed = mouse_btns[2]  # Right click
            
            # Populate a single 'Right' hand with virtual landmarks
            v_lms = generate_virtual_landmarks(mx, my, pinch_pressed, curl_pressed, virtual_rotation)
            
            # Simulate MediaPipe container format
            virtual_hands_data = {
                'Right': {
                    'landmarks': v_lms,
                    'confidence': 1.0,
                    'active': True
                },
                'Left': {
                    'landmarks': np.zeros((21, 3), dtype=np.float32),
                    'confidence': 0.0,
                    'active': False
                }
            }
            gestures_data = analyzer.process(virtual_hands_data, dt)
            hands_active_count = 1
        else:
            # Use real camera tracker data
            gestures_data = analyzer.process(raw_hands, dt)
            for side in ['Left', 'Right']:
                if gestures_data[side]['active']:
                    hands_active_count += 1

        # 3. Handle screen shake state from physics triggers
        if physics.shake_trigger > 0.0:
            shake_timer = 0.35  # Shake duration 350ms
            shake_intensity = physics.shake_trigger
            physics.shake_trigger = 0.0
            
        dx = 0
        dy = 0
        if shake_timer > 0.0:
            shake_timer -= dt
            dx = int(np.random.uniform(-shake_intensity, shake_intensity))
            dy = int(np.random.uniform(-shake_intensity, shake_intensity))
            shake_intensity = max(0.0, shake_intensity - dt * 25.0)

        # 4. Apply Void Warp effect (Space bending) on camera background BGR
        if physics.mode == 3 and physics.transition_timer <= 0.0:
            # Apply warp centered around fists
            for side in ['Left', 'Right']:
                hand = gestures_data[side]
                if hand['active']:
                    avg_curl = np.mean(hand['curls'][1:])
                    if avg_curl > 0.8:  # Fist
                        fx, fy = hand['palm_center']
                        processed_bg = warper.apply(processed_bg, fx, fy)
                        
            # Apply warp at singularity spot if active
            if physics.singularity_active and physics.singularity_stage == 0:
                sx, sy = physics.singularity_pos
                # Pulse warp intensity during implosion
                pulse_int = 0.5 + 0.3 * math.sin(time_val * 35.0)
                warper.intensity = pulse_int
                processed_bg = warper.apply(processed_bg, sx, sy)
                # Reset intensity to default
                warper.intensity = 0.45
            
            # Gojo Infinity: subtle space warp at midpoint between hands
            if gestures_data.get('gojo_infinity', False):
                lh = gestures_data['Left']
                rh = gestures_data['Right']
                if lh['active'] and rh['active']:
                    gx = (lh['palm_center'][0] + rh['palm_center'][0]) * 0.5
                    gy = (lh['palm_center'][1] + rh['palm_center'][1]) * 0.5
                    warper.intensity = 0.28 + 0.12 * math.sin(time_val * 7.0)
                    processed_bg = warper.apply(processed_bg, gx, gy)
                    warper.intensity = 0.45

        # 5. Update physics particles
        physics.update(gestures_data, dt)
        
        # 6. Render Foreground (particles, hand overlay, lightning onto a black canvas)
        foreground_surf.fill((0, 0, 0))
        
        # Draw Particles with Motion Blur Trails
        active_indices = np.where(physics.active)[0]
        for idx in active_indices:
            px, py = physics.pos[idx]
            ppx, ppy = physics.prev_pos[idx]
            size = physics.size[idx]
            color = physics.color[idx]
            alpha = physics.alpha[idx]
            
            # If moving fast, draw a line for motion blur, else draw a circle
            speed_sq = (px - ppx)**2 + (py - ppy)**2
            if speed_sq > 25.0:
                pygame.draw.line(foreground_surf, (*color, int(alpha * 0.85)), (int(ppx), int(ppy)), (int(px), int(py)), max(1, int(size)))
            else:
                pygame.draw.circle(foreground_surf, (*color, int(alpha)), (int(px), int(py)), max(1, int(size)))

        # Draw Electric Lightning Arcs (Mode 5)
        lightning_arcs = physics.get_lightning_arcs(gestures_data)
        if len(lightning_arcs) > 0:
            draw_electricity_arcs(foreground_surf, lightning_arcs, time_val)

        # Draw Technical Hand Overlays
        draw_hand_overlay(foreground_surf, gestures_data, time_val, physics)

        # 7. Apply Post-processing (OpenCV Bloom) & Compositing
        # Convert Pygame foreground surface to RGB numpy array
        fg_rgb_data = pygame.image.tostring(foreground_surf, 'RGB')
        fg_np = np.frombuffer(fg_rgb_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        
        # Compositing step (desaturated camera BGR + bloomed foreground RGB -> final RGB)
        composite_rgb = effects.apply_bloom_composite(processed_bg, fg_np)
        
        # Apply Chromatic Aberration during shake
        if shake_timer > 0.0 and shake_intensity > 2.0:
            aberration_shift = int(shake_intensity * 0.35)
            composite_rgb = effects.apply_chromatic_aberration(composite_rgb, aberration_shift, aberration_shift // 2)
            
        # Convert composite back to Pygame surface
        composite_surf = pygame.image.frombuffer(composite_rgb.tobytes(), (WIDTH, HEIGHT), 'RGB')
        screen.blit(composite_surf, (dx, dy))
        
        # 8. Draw HUD (drawn on top of everything without bloom to stay crisp and sharp)
        draw_hud(screen, physics, clock.get_fps(), hands_active_count, gestures_data, virtual_hand_active, show_help)
        
        # Flip frame buffer
        pygame.display.flip()

    # Clean cleanup
    tracker.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

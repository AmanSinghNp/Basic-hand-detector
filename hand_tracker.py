import cv2
import mediapipe as mp
import math
import numpy as np
import random
from collections import deque

# --- Particle System ---
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = 1.0  # Life from 1.0 to 0.0
        self.decay = random.uniform(0.02, 0.05)
        self.color = color # Tuple (B, G, R)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        if self.life > 0:
            alpha = int(self.life * 255)
            # Create a localized overlay for transparency could be expensive per particle.
            # Simple circle with fading radius/color.
            radius = int(3 * self.life) + 1
            # Adjust color brightness by life
            c = tuple(int(ch * self.life) for ch in self.color)
            cv2.circle(img, (int(self.x), int(self.y)), radius, c, -1)

# --- Global Trails ---
# Dictionary to store deque for trails: keys 'thumb', 'index'
trails = {
    'thumb': deque(maxlen=20),
    'index': deque(maxlen=20)
}

# --- EMA Smoothing ---
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def is_finger_extended(landmarks, tip_id, pip_id):
    """
    Checks if a finger is extended by comparing Tip Y with PIP Y.
    Note: Y increases downwards in image coordinates.
    So Tip < PIP means extended (upwards).
    This logic assumes hand is upright. For general cases, angle checks are better,
    but this suffices for simple "palm facing camera" gestures.
    """
    return landmarks[tip_id].y < landmarks[pip_id].y

def get_hsv_color(hue_value):
    """
    Returns BGR color tuple from a Hue value (0-179).
    """
    # Create 1x1 pixel with HSV color
    hsv_pixel = np.uint8([[[hue_value, 255, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    return tuple(map(int, bgr_pixel[0][0]))

def apply_neon_glow(image, mask, glow_intensity):
    """
    Applies a gaussian blur to the mask and adds it to the image.
    :param image: Main BGR image
    :param mask: Black image with only bright colored edges drawn
    :param glow_intensity: Sigma for Gaussian Blur
    """
    if glow_intensity <= 0:
        return image
    
    # Kernel size must be odd
    ksize = (glow_intensity * 2 + 1, glow_intensity * 2 + 1)
    blurred = cv2.GaussianBlur(mask, ksize, 0)
    
    # Add weighted: Image + Blurred_Mask
    return cv2.addWeighted(image, 1.0, blurred, 1.0, 0)

def draw_cube(mask_layer, center_point, size, camera_matrix, dist_coeffs, 
              rot_x=0, rot_y=0, rot_z=0, 
              fill_faces=False, face_color=(100, 100, 100), edge_color_base=None):
    """
    Draws a 3D wireframe cube on the MASK layer (for glow).
    Accepts 3-axis rotation (Euler angles).
    """
    s = size / 2.0
    
    vertices = np.array([
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]
    ], dtype=np.float32)

    # Rotation Logic (3D)
    cx, sx = math.cos(rot_x), math.sin(rot_x)
    cy, sy = math.cos(rot_y), math.sin(rot_y)
    cz, sz = math.cos(rot_z), math.sin(rot_z)
    
    # Rotation Matrices
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    
    # Combined Rotation: Rz * Ry * Rx
    R_combined = Rz @ Ry @ Rx
    
    vertices = np.dot(vertices, R_combined.T)

    # Projection setup
    Z_depth = 100.0
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    X_3d = (center_point[0] - cx) * Z_depth / fx
    Y_3d = (center_point[1] - cy) * Z_depth / fy
    
    tvec = np.array([X_3d, Y_3d, Z_depth], dtype=np.float32)
    rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    img_points, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2).astype(int)
    
    # Define Faces (vertex indices)
    faces = [
        [0, 1, 2, 3], # Front
        [4, 5, 6, 7], # Back
        [0, 1, 5, 4], # Top
        [2, 3, 7, 6], # Bottom
        [0, 3, 7, 4], # Left
        [1, 2, 6, 5]  # Right
    ]

    # Draw Filled Faces
    if fill_faces:
         for face in faces:
            pts = np.array([img_points[i] for i in face])
            cv2.fillPoly(mask_layer, [pts], face_color)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Futuristic Colors
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    neon_green = (0, 255, 100)
    
    vertex_list = [] 

    for i, edge in enumerate(edges):
        pt1 = tuple(img_points[edge[0]])
        pt2 = tuple(img_points[edge[1]])
        
        if edge_color_base:
            color = edge_color_base
        else:
            if i < 4: color = cyan
            elif i < 8: color = magenta
            else: color = neon_green
        
        cv2.line(mask_layer, pt1, pt2, color, 3)
        vertex_list.append(pt1)
        vertex_list.append(pt2)

    return vertex_list

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    print("Scanning for available cameras...")
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras found.")
        return None
        
    if len(cameras) == 1:
        print(f"Only one camera found (Index {cameras[0]}). Using it automatically.")
        return cameras[0]
        
    print("\nAvailable Cameras:")
    for idx in cameras:
        print(f" - Camera Index: {idx}")
        
    while True:
        try:
            choice = input("\nEnter the camera index you want to use: ")
            choice = int(choice)
            if choice in cameras:
                return choice
            else:
                print("Invalid index. Please choose from the list.")
        except ValueError:
            print("Please enter a valid number.")

def nothing(x):
    pass

def main():
    camera_index = select_camera()
    if camera_index is None:
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, # Limit to 1 hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create Trackbar Window
    cv2.namedWindow('Controls')
    cv2.resizeWindow('Controls', 300, 150)
    cv2.createTrackbar('Glow Intensity', 'Controls', 15, 50, nothing)

    print("Hand tracker started. Press 'q' to exit.")

    fingertip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
    # Indices for checking extension (Tip vs PIP)
    finger_indices = {
        'thumb': (4, 3), # Thumb is special, compare tip x/y vs IP. using IP(3)
        'index': (8, 6),
        'middle': (12, 10),
        'ring': (16, 14),
        'pinky': (20, 18)
    }

    frame_counter = 0
    particles = []
    
    # Initialize EMA filters
    ema_center_x = EMAFilter(alpha=0.3)
    ema_center_y = EMAFilter(alpha=0.3)
    ema_size = EMAFilter(alpha=0.2)
    ema_rot_x = EMAFilter(alpha=0.1)
    ema_rot_y = EMAFilter(alpha=0.1)
    ema_rot_z = EMAFilter(alpha=0.1)
    
    # Pinch resizing state
    cube_size = 100.0 # Default locked size
    is_pinching = False
    pinch_threshold = 40 # Pixels

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, c = image.shape
        glow_mask = np.zeros_like(image)
        
        # Trackbar Values
        glow_val = cv2.getTrackbarPos('Glow Intensity', 'Controls')

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array(
             [[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]], dtype = "double"
        )
        dist_coeffs = np.zeros((4,1))

        if results.multi_hand_landmarks:
            # Process only the FIRST hand
            hand_landmarks = results.multi_hand_landmarks[0]
            lms = hand_landmarks.landmark
            
            # --- Hand Orientation Logic (Palm Normal) ---
            wrist = lms[0]
            index_mcp = lms[5]
            pinky_mcp = lms[17]
            
            # Vectors: Wrist to Index, Wrist to Pinky
            v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
            v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
            
            # Palm Normal (Cross Product)
            palm_normal = np.cross(v1, v2)
            norm_mag = np.linalg.norm(palm_normal)
            if norm_mag != 0:
                palm_normal /= norm_mag

            # Convert to Euler Angles (Approximate)
            # Yaw (rotation around Y - shaking head "no") -> related to X component of normal?
            # Pitch (rotation around X - nodding "yes") -> related to Y component
            # Roll (rotation around Z - tilting head) -> related to orientation of vector v1 vs horizon
            
            # Simple mapping:
            # Pitch (X-axis rot) -> Tilt hand up/down (Y component of normal)
            target_rot_x = math.atan2(palm_normal[1], palm_normal[2])
            # Yaw (Y-axis rot) -> Turn hand left/right (X component of normal)
            target_rot_y = math.atan2(-palm_normal[0], palm_normal[2])
            # Roll (Z-axis rot) -> Tilt hand side-to-side (Angle of Index-Pinky line)
            v_across = np.array([pinky_mcp.x - index_mcp.x, pinky_mcp.y - index_mcp.y])
            target_rot_z = math.atan2(v_across[1], v_across[0])
            
            # Smooth Rotations
            smooth_rot_x = ema_rot_x.update(target_rot_x)
            smooth_rot_y = ema_rot_y.update(target_rot_y)
            smooth_rot_z = ema_rot_z.update(target_rot_z)

            # --- Multi-Finger Gesture Logic ---
            is_extended = {}
            for name, (tip, pip) in finger_indices.items():
                is_extended[name] = lms[tip].y < lms[pip].y
            
            # 1. Middle Finger: Transparency
            fill_faces = not is_extended['middle']
            
            # 2. Ring Finger: Color Shift
            edge_color = None
            if not is_extended['ring']:
                frame_counter += 1
                hue = int((frame_counter * 2) % 180)
                edge_color = get_hsv_color(hue)
            
            # 3. Pinky: Particle Toggle
            particles_enabled = is_extended['pinky']

            # --- Visual Feedback Rings ---
            for name, (tip, _) in finger_indices.items():
                lm_tip = lms[tip]
                tx, ty = int(lm_tip.x * w), int(lm_tip.y * h)
                color = (0, 255, 0) if is_extended[name] else (0, 0, 255)
                cv2.circle(image, (tx, ty), 5, color, 1)

            # --- AR Cube Logic (Pinch to Resize) ---
            thumb = lms[4]
            index = lms[8]
            thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
            index_x, index_y = int(index.x * w), int(index.y * h)
            
            trails['thumb'].append((thumb_x, thumb_y))
            trails['index'].append((index_x, index_y))
            
            raw_cx = (thumb_x + index_x) // 2
            raw_cy = (thumb_y + index_y) // 2
            
            # Check Pinch Distance
            pinch_dist = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # Pinch State Logic
            if pinch_dist < pinch_threshold:
                if not is_pinching:
                    is_pinching = True # Start pinching
                # While pinching, update size based on distance? 
                # Actually, "Pinch to set size" usually means:
                # Pinch -> Hold -> Release. 
                # OR: "Two hand pinch" = scale. 
                # Single hand pinch usually means "Grab".
                # Let's implement: Pinch = Set size to minimum? No, that's bad.
                # Let's implement: Distance between Thumb/Index sets size continuously, 
                # BUT we clamp it or lock it?
                # The user asked: "Pinch-to-Set Size".
                # Interpretation: The distance *IS* the size.
                # BUT we want to 'lock' it when released?
                # Let's stick to the previous behavior: Distance = Size.
                # It is intuitive. 
                # To "Lock", maybe we use the Index+Middle gesture from before?
                # Let's just update the size based on distance continuously for now as it's most robust.
                cube_size = pinch_dist * 0.5
            else:
                 is_pinching = False
                 # Size remains whatever it was last frame if we don't update it.
                 # Actually, let's allow it to always scale with fingers for now.
                 cube_size = pinch_dist * 0.5

            smooth_cx = int(ema_center_x.update(raw_cx))
            smooth_cy = int(ema_center_y.update(raw_cy))
            smooth_size = ema_size.update(cube_size)

            if smooth_size > 5:
                cube_vertices = draw_cube(
                    mask_layer=glow_mask, 
                    center_point=(smooth_cx, smooth_cy), 
                    size=smooth_size, 
                    camera_matrix=camera_matrix, 
                    dist_coeffs=dist_coeffs, 
                    rot_x=smooth_rot_x,
                    rot_y=smooth_rot_y,
                    rot_z=smooth_rot_z,
                    fill_faces=fill_faces,
                    face_color=(0, 50, 50) if fill_faces else None, 
                    edge_color_base=edge_color
                )
                
                if particles_enabled and cube_vertices:
                    v = random.choice(cube_vertices)
                    p_color = edge_color if edge_color else ((255, 255, 0) if random.random() > 0.5 else (255, 0, 255))
                    particles.append(Particle(v[0], v[1], p_color))

        # --- Draw Trails ---
        for key in trails:
            pts = list(trails[key])
            for i in range(1, len(pts)):
                thickness = int(math.sqrt(i))
                cv2.line(glow_mask, pts[i-1], pts[i], (0, 255, 255), thickness)

        # --- Update & Draw Particles ---
        particles = [p for p in particles if p.update()]
        for p in particles:
            p.draw(glow_mask)

        # --- Apply Glow & Combine ---
        image = cv2.add(image, glow_mask)
        final_image = apply_neon_glow(image, glow_mask, glow_val)

        # --- Legend ---
        cv2.putText(final_image, "Pinky: Particles", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_image, "Ring (Fold): Color Cycle", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_image, "Middle (Fold): Solid Face", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_image, "Hand Tilt: Rotate Cube", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Hand Tracker - Fingertip Boxes & AR Cube', final_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

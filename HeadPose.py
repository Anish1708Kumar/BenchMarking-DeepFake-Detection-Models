import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# --- Paths ---
REAL_DIR = 'UADFV/real/frames/'
FAKE_DIR = 'UADFV/fake/frames/'
DLIB_LANDMARK_PATH = 'shape_predictor_68_face_landmarks.dat'


# --- Dlib setup ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)

# --- Head pose estimation ---
def get_head_pose(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    if len(dets) == 0:
        return None
    shape = predictor(gray, dets[0])
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corner
        (shape.part(48).x, shape.part(48).y),  # Left mouth corner
        (shape.part(54).x, shape.part(54).y)   # Right mouth corner
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    size = img.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None
    rot_mat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    pitch = np.arctan2(-rot_mat[2, 0], sy) * 180 / np.pi
    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0]) * 180 / np.pi
    roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2]) * 180 / np.pi
    return [yaw, pitch, roll]

# --- Image loader ---
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read {path}")
    return img

# --- Feature extraction ---
def extract_features_from_directory(directory, label):
    features = []
    labels = []
    for fname in tqdm(os.listdir(directory)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        path = os.path.join(directory, fname)
        img = load_image(path)
        if img is None:
            continue
        pose = get_head_pose(img)
        if pose is not None:
            features.append(pose)
            labels.append(label)
    return features, labels

# --- Main pipeline ---
# Extract features
print("Processing real images...")
X_real, y_real = extract_features_from_directory(REAL_DIR, 0)
print("Processing fake images...")
X_fake, y_fake = extract_features_from_directory(FAKE_DIR, 1)

# Combine and shuffle
X = X_real + X_fake
y = y_real + y_fake
print(f"Total samples: {len(X)}")

if len(X) == 0:
    print("No head pose features extracted. Check your image paths and face detector.")
    exit()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_pred)
print(f"AUC (Area Under ROC Curve): {auc:.4f}")

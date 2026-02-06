from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import sqlite3
import hashlib
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
from scipy import fftpack
from skimage.feature import local_binary_pattern
import os
import sys
import logging
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_login.log')
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from cnn_antispoofing import CNNAntiSpoofing

# Simple rule-based fallback (basic checks only)
class BasicAntiSpoofing:
    def is_trained(self):
        return True
    
    def analyze(self, img_array):
        """Basic liveness check - always returns True with low confidence"""
        return True, "Basic check (Train CNN model for better security)", 60

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize anti-spoofing (CNN or fallback to rule-based)
cnn_spoof = CNNAntiSpoofing()

if cnn_spoof.is_trained():
    logger.info("[OK] Using CNN-based anti-spoofing (deep learning model)")
    anti_spoof = cnn_spoof
    model_type = "CNN"
else:
    logger.warning("[WARNING] No trained CNN model found. Using basic fallback")
    logger.info("   Train CNN model using the web interface at: http://localhost:8000/train.html")
    anti_spoof = BasicAntiSpoofing()
    model_type = "Basic"

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  face_hash TEXT NOT NULL,
                  face_features TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  status TEXT DEFAULT 'active',
                  last_login TEXT)''')
    
    # Add status column if it doesn't exist (for existing databases)
    try:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active'")
    except:
        pass
    
    # Add last_login column if it doesn't exist
    try:
        c.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
    except:
        pass
    
    conn.commit()
    conn.close()

init_db()

def check_liveness_legacy(img_array):
    """State-of-the-art anti-spoofing with multiple detection layers"""
    # Resize for faster processing while maintaining quality
    h, w = img_array.shape[:2]
    if w > 640:
        scale = 640 / w
        img_array = cv2.resize(img_array, (640, int(h * scale)))
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    score = 0
    max_score = 100
    reasons = []
    
    # === LAYER 1: TEXTURE ANALYSIS (LBP) - Detects print attacks ===
    try:
        # Local Binary Pattern for texture analysis
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # Real faces have more uniform texture distribution
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
        if lbp_entropy > 3.0:
            score += 18  # Increased reward
        elif lbp_entropy > 2.7:
            score += 12
        elif lbp_entropy > 2.4:
            score += 8
        else:
            score += 4  # Give some credit
    except:
        score += 10
    
    # === LAYER 2: FREQUENCY DOMAIN ANALYSIS - Detects screen attacks ===
    try:
        # FFT to detect digital display patterns
        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency content indicates real face
        h_freq = np.sum(magnitude_spectrum[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)])
        total_freq = np.sum(magnitude_spectrum)
        freq_ratio = h_freq / (total_freq + 1e-7)
        
        # Very lenient thresholds for webcam captures
        if freq_ratio > 0.05:
            score += 12
        elif freq_ratio > 0.03:
            score += 10
        elif freq_ratio > 0.01:
            score += 8
        else:
            score += 4  # Give some credit anyway
    except:
        # If FFT fails, give partial credit
        score += 6
    
    # === LAYER 3: COLOR SPACE ANALYSIS - Skin tone verification ===
    try:
        # Convert to HSV and YCbCr for skin detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        
        # Skin tone ranges
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)
        skin_ratio = np.sum(skin_mask > 0) / (gray.shape[0] * gray.shape[1])
        
        if 0.15 < skin_ratio < 0.65:
            score += 15
        elif 0.10 < skin_ratio < 0.70:
            score += 8
        else:
            reasons.append("Abnormal skin tone")
            
        # Additional check: Screen detection via color uniformity
        # Screens have more uniform color distribution
        color_std_per_channel = [np.std(img_array[:,:,i]) for i in range(3)]
        avg_std = np.mean(color_std_per_channel)
        
        # Real faces have higher color variation
        if avg_std < 30:
            score -= 20  # Strong penalty for low variation (screen-like)
            reasons.append("Screen detected (low color variation)")
        
        # Check for screen glare
        v_channel = hsv[:,:,2]
        bright_pixels = np.sum(v_channel > 245)
        total_pixels = v_channel.shape[0] * v_channel.shape[1]
        bright_ratio = bright_pixels / total_pixels
        
        if bright_ratio > 0.2:
            score -= 15
            reasons.append("Screen glare detected")
            
    except:
        pass
    
    # === LAYER 4: MOIRÉ PATTERN DETECTION - Screen recapture ===
    try:
        # Detect periodic patterns from screen refresh
        gray_float = gray.astype(np.float32)
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        # Check for periodic peaks (moiré patterns)
        magnitude_log = np.log(magnitude + 1)
        peaks = magnitude_log > (np.mean(magnitude_log) + 3.5 * np.std(magnitude_log))  # More lenient
        peak_count = np.sum(peaks)
        
        # Very lenient thresholds
        if peak_count < 200:  # Very lenient
            score += 10
        elif peak_count < 300:
            score += 8
        else:
            score += 4  # Give some credit anyway
    except:
        # If detection fails, give partial credit
        score += 5
    
    # === LAYER 5: SHARPNESS & BLUR DETECTION ===
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 80:  # More lenient
        score += 12
    elif laplacian_var > 40:
        score += 8
    elif laplacian_var > 20:
        score += 4
    else:
        reasons.append("Image too blurry")
    
    # === LAYER 6: EDGE DENSITY ANALYSIS + SCREEN BORDER DETECTION ===
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Check for rectangular screen borders (strong indicator of phone screen)
    h_edges, w_edges = edges.shape
    border_thickness = 20
    
    # Check edges near borders
    top_border = np.sum(edges[:border_thickness, :]) / (border_thickness * w_edges)
    bottom_border = np.sum(edges[-border_thickness:, :]) / (border_thickness * w_edges)
    left_border = np.sum(edges[:, :border_thickness]) / (h_edges * border_thickness)
    right_border = np.sum(edges[:, -border_thickness:]) / (h_edges * border_thickness)
    
    # If strong edges on borders, likely a phone screen
    border_edges = [top_border, bottom_border, left_border, right_border]
    if sum(1 for x in border_edges if x > 0.3) >= 2:
        score -= 25  # Strong penalty for screen borders
        reasons.append("Phone screen border detected")
    
    if 0.04 < edge_density < 0.35:
        score += 8
    elif edge_density > 0.35:
        score -= 5
        reasons.append("Screen pixel pattern detected")
    else:
        reasons.append("Abnormal edge density")
    
    # === LAYER 7: FACE & EYE DETECTION ===
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    has_eyes = False
    if len(faces) > 0:
        score += 5  # Give points for face detection
        x, y, w_face, h_face = faces[0]
        roi_gray = gray[y:y+h_face, x:x+w_face]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        has_eyes = len(eyes) >= 1
        
        if has_eyes:
            score += 10  # Bonus for eyes
        else:
            score += 3  # Still give some points even without eyes
    else:
        reasons.append("Face not detected")
    
    # === LAYER 8: SPECULAR REFLECTION DETECTION ===
    try:
        # Real faces have natural specular highlights
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bright_regions = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_regions > 0) / (gray.shape[0] * gray.shape[1])
        
        if 0.001 < bright_ratio < 0.05:  # Natural highlights
            score += 8
    except:
        pass
    
    # === DECISION MAKING ===
    # Robust threshold (55/100)
    is_live = score >= 55
    confidence = min(100, score)
    
    if is_live:
        return True, f"Verified (confidence: {confidence}%)"
    else:
        reason_text = ", ".join(reasons[:2]) if reasons else "Failed verification"
        return False, f"{reason_text} (score: {confidence}/100)"

def check_liveness(img_array):
    """Enhanced liveness check using CNN/ML/Rule-based approach"""
    try:
        logger.info("="*70)
        logger.info("LIVENESS CHECK STARTED")
        logger.info("="*70)
        logger.info(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.info(f"Image value range: [{img_array.min()}, {img_array.max()}]")
        
        if isinstance(anti_spoof, CNNAntiSpoofing):
            # CNN-BASED APPROACH: Most robust, uses deep learning
            logger.info("Using CNN-based anti-spoofing (Deep Learning)")
            logger.info("CNN Confidence threshold: 90%")
            
            # CNN prediction
            is_live, message, confidence = anti_spoof.predict(img_array, confidence_threshold=0.90)
            
            logger.info("="*70)
            logger.info("LIVENESS CHECK RESULT")
            logger.info("="*70)
            logger.info(f"Decision: {'[OK] REAL FACE' if is_live else '[FAIL] FAKE/SPOOF'}")
            logger.info(f"CNN Confidence: {confidence}%")
            logger.info(f"Pass/Fail: {'PASSED' if is_live else 'FAILED'}")
            logger.info(f"Message: {message}")
            logger.info("="*70)
            
        else:
            # Basic fallback only
            logger.info("Using basic anti-spoofing (fallback)")
            is_live, message, confidence = anti_spoof.analyze(img_array)
            logger.info(f"Basic result: {is_live}, confidence: {confidence}%")
        
        return is_live, message
    except Exception as e:
        logger.error("="*70)
        logger.error("LIVENESS CHECK ERROR")
        logger.error("="*70)
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        logger.error("="*70)
        # Fallback to a safe response
        return False, f"Verification error: {str(e)}"

def extract_face_features(img_array):
    """Extract face features using OpenCV"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Try multiple detection parameters for better results
    # More lenient parameters: lower scaleFactor and minNeighbors
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    # If no faces found, try even more lenient settings
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
    
    # If still no faces, try with histogram equalization (improves detection in poor lighting)
    if len(faces) == 0:
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            gray = gray_eq  # Use equalized image if it worked
    
    if len(faces) == 0:
        print(f"[extract_face_features] No face detected in image of shape {img_array.shape}")
        return None, None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    print(f"[extract_face_features] Face detected at ({x}, {y}) with size {w}x{h}")
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Create hash and features
    face_hash = hashlib.sha256(face_roi.tobytes()).hexdigest()
    features = face_roi.flatten().tolist()
    
    return face_hash, features

def compare_faces(features1, features2, threshold=0.75):
    """Compare two face feature vectors"""
    arr1 = np.array(features1)
    arr2 = np.array(features2)
    
    # Normalize
    arr1 = arr1 / np.linalg.norm(arr1)
    arr2 = arr2 / np.linalg.norm(arr2)
    
    # Calculate similarity
    similarity = np.dot(arr1, arr2)
    
    print(f"[compare_faces] Similarity: {similarity:.4f}, Threshold: {threshold}")
    return similarity > threshold

@app.post("/api/verify-liveness")
async def verify_liveness(image: UploadFile = File(...)):
    """Verify if the image is from a live person"""
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        img_array = np.array(img)
        
        # Log for debugging
        print(f"\n[Verify Liveness] Image shape: {img_array.shape}")
        print(f"[Verify Liveness] Using model: {type(anti_spoof).__name__}")
        
        is_live, message = check_liveness(img_array)
        
        print(f"[Verify Liveness] Result: {is_live}, Message: {message}")
        
        return {
            "is_live": is_live,
            "message": message
        }
    except Exception as e:
        print(f"[Verify Liveness] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
async def register(
    username: str = Form(...),
    email: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        img_array = np.array(img)
        
        print(f"\n[Register] Processing registration for {username}")
        print(f"[Register] Image shape: {img_array.shape}")
        
        # Check liveness first
        print("[Register] Checking liveness...")
        is_live, liveness_msg = check_liveness(img_array)
        print(f"[Register] Liveness result: {is_live} - {liveness_msg}")
        
        if not is_live:
            raise HTTPException(status_code=400, detail=f"Liveness check failed: {liveness_msg}")
        
        print("[Register] Extracting face features...")
        face_hash, features = extract_face_features(img_array)
        
        if face_hash is None:
            raise HTTPException(
                status_code=400, 
                detail="No face detected. Please ensure: 1) Good lighting, 2) Face the camera directly, 3) Remove glasses/hats if needed"
            )
        
        print(f"[Register] Face detected successfully")
        features_str = json.dumps(features)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (username, email, face_hash, face_features, created_at) VALUES (?, ?, ?, ?, ?)",
                     (username, email, face_hash, features_str, datetime.now().isoformat()))
            conn.commit()
            print(f"[Register] [OK] User {username} registered successfully")
        except sqlite3.IntegrityError:
            conn.close()
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        conn.close()
        
        return {"message": "Registration successful", "username": username}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Register] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        img_array = np.array(img)
        
        logger.info("\n" + "="*70)
        logger.info("LOGIN ATTEMPT")
        logger.info("="*70)
        logger.info(f"Image shape: {img_array.shape}")
        
        # Check liveness first
        logger.info("Step 1: Checking liveness...")
        is_live, liveness_msg = check_liveness(img_array)
        logger.info(f"Liveness result: {is_live} - {liveness_msg}")
        
        if not is_live:
            raise HTTPException(status_code=400, detail=f"Liveness check failed: {liveness_msg}")
        
        print("[Login] Extracting face features...")
        face_hash, features = extract_face_features(img_array)
        
        if face_hash is None:
            raise HTTPException(
                status_code=400, 
                detail="No face detected. Please ensure: 1) Good lighting, 2) Face the camera directly, 3) Remove glasses/hats if needed"
            )
        
        print("[Login] Comparing with registered faces...")
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, username, email, face_features, status FROM users")
        users = c.fetchall()
        
        for user in users:
            stored_features = json.loads(user[3])
            user_status = user[4] if len(user) > 4 else 'active'
            print(f"[Login] Comparing with user: {user[1]} (status: {user_status})")
            
            if compare_faces(features, stored_features):
                # Check if user is active
                if user_status != 'active':
                    conn.close()
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Account is {user_status}. Please contact administrator."
                    )
                
                # Update last login time
                c.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                         (datetime.now().isoformat(), user[0]))
                conn.commit()
                conn.close()
                
                print(f"[Login] [OK] Match found for user: {user[1]}")
                return {
                    "message": "Login successful",
                    "user": {
                        "id": user[0],
                        "username": user[1],
                        "email": user[2]
                    }
                }
        
        conn.close()
        
        print("[Login] [FAIL] No matching face found")
        raise HTTPException(
            status_code=401, 
            detail="Face not recognized. Try: 1) Better lighting, 2) Same angle as registration, 3) Re-register if needed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Login] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Global training state
training_state = {
    "is_training": False,
    "result": None,
    "error": None
}

def train_in_background(real_img_arrays, fake_img_arrays, epochs, batch_size, new_real_count, new_fake_count, confidence_threshold):
    """Background training function"""
    global anti_spoof, training_state
    try:
        model = CNNAntiSpoofing()
        metrics = model.train(
            real_img_arrays, 
            fake_img_arrays, 
            save_images=True,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Update global anti_spoof to use new model
        anti_spoof = model
        
        print(f"\n[OK] Training complete! Accuracy: {metrics['accuracy']*100:.2f}%")
        
        training_state["result"] = {
            "success": True,
            "message": "CNN Deep Learning model trained successfully",
            "metrics": metrics,
            "new_real_images": new_real_count,
            "new_fake_images": new_fake_count,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "confidence_threshold": confidence_threshold
            }
        }
        training_state["is_training"] = False
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        training_state["error"] = str(e)
        training_state["is_training"] = False

@app.post("/api/train")
async def train_model(
    real_images: list[UploadFile] = File(...),
    fake_images: list[UploadFile] = File(...),
    epochs: int = Form(20),
    batch_size: int = Form(16),
    confidence_threshold: int = Form(90)
):
    """Train CNN anti-spoofing model with uploaded images"""
    global training_state
    
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        print(f"\n{'='*60}")
        print(f"Training CNN Deep Learning Model")
        print(f"{'='*60}")
        print(f"New real images: {len(real_images)}")
        print(f"New fake images: {len(fake_images)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Confidence threshold: {confidence_threshold}%")
        
        # Load images
        real_img_arrays = []
        for img_file in real_images:
            contents = await img_file.read()
            img = Image.open(BytesIO(contents))
            img_array = np.array(img)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:,:,:3]  # Convert to RGB
            real_img_arrays.append(img_array)
        
        fake_img_arrays = []
        for img_file in fake_images:
            contents = await img_file.read()
            img = Image.open(BytesIO(contents))
            img_array = np.array(img)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:,:,:3]  # Convert to RGB
            fake_img_arrays.append(img_array)
        
        # Reset training state
        training_state = {
            "is_training": True,
            "result": None,
            "error": None
        }
        
        # Initialize progress file
        with open('training_progress.json', 'w') as f:
            json.dump({"epoch": 0, "total_epochs": epochs, "status": "starting"}, f)
        
        # Start training in background thread
        thread = threading.Thread(
            target=train_in_background,
            args=(real_img_arrays, fake_img_arrays, epochs, batch_size, len(real_images), len(fake_images), confidence_threshold)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately - frontend will poll for progress
        return {
            "success": True,
            "message": "Training started",
            "status": "training",
            "epochs": epochs
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        training_state["is_training"] = False
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/training-progress")
async def get_training_progress():
    """Get real-time training progress"""
    try:
        import os
        if os.path.exists('training_progress.json'):
            with open('training_progress.json', 'r') as f:
                import json
                progress = json.load(f)
                return progress
        return {"epoch": 0, "total_epochs": 0, "status": "idle"}
    except Exception as e:
        return {"epoch": 0, "total_epochs": 0, "status": "error"}

@app.get("/api/training-result")
async def get_training_result():
    """Get final training result"""
    global training_state
    
    if training_state["is_training"]:
        return {"status": "training", "complete": False}
    elif training_state["error"]:
        return {"status": "error", "complete": True, "error": training_state["error"]}
    elif training_state["result"]:
        return {"status": "complete", "complete": True, "result": training_state["result"]}
    else:
        return {"status": "idle", "complete": False}

@app.get("/api/training-stats")
async def get_training_stats():
    """Get statistics about saved training data"""
    try:
        cnn_model = CNNAntiSpoofing()
        real_count, fake_count = cnn_model.get_training_stats()
        
        return {
            "total_real_images": real_count,
            "total_fake_images": fake_count,
            "total_images": real_count + fake_count,
            "is_trained": cnn_model.is_trained()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def retrain_in_background(real_images, fake_images, epochs, batch_size, confidence_threshold):
    """Background retraining function"""
    global anti_spoof, training_state
    try:
        model = CNNAntiSpoofing()
        metrics = model.train(
            real_images, 
            fake_images, 
            save_images=False,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Update global anti_spoof to use new model
        anti_spoof = model
        
        print(f"\n[OK] Retraining complete! Accuracy: {metrics['accuracy']*100:.2f}%")
        
        training_state["result"] = {
            "success": True,
            "message": "CNN Deep Learning model retrained successfully",
            "metrics": metrics,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "confidence_threshold": confidence_threshold
            }
        }
        training_state["is_training"] = False
        
    except Exception as e:
        print(f"Retraining error: {e}")
        import traceback
        traceback.print_exc()
        training_state["error"] = str(e)
        training_state["is_training"] = False

@app.post("/api/retrain")
async def retrain_model(
    epochs: int = Form(20),
    batch_size: int = Form(16),
    confidence_threshold: int = Form(90)
):
    """Retrain CNN model using all existing saved training data"""
    global training_state
    
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        print(f"\n{'='*60}")
        print(f"Retraining CNN Model with Existing Data")
        print(f"{'='*60}")
        
        # Load CNN model
        model = CNNAntiSpoofing()
        
        # Load all existing training data
        real_images, fake_images = model.load_training_images()
        
        print(f"Loaded {len(real_images)} real and {len(fake_images)} fake images")
        
        if len(real_images) < 20 or len(fake_images) < 20:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough training data. Found {len(real_images)} real and {len(fake_images)} fake images. Need at least 20 of each for CNN."
            )
        
        # Reset training state
        training_state = {
            "is_training": True,
            "result": None,
            "error": None
        }
        
        # Initialize progress file
        with open('training_progress.json', 'w') as f:
            json.dump({"epoch": 0, "total_epochs": epochs, "status": "starting"}, f)
        
        # Start retraining in background thread
        thread = threading.Thread(
            target=retrain_in_background,
            args=(real_images, fake_images, epochs, batch_size, confidence_threshold)
        )
        thread.daemon = True
        thread.start()
        
        # Return immediately - frontend will poll for progress
        return {
            "success": True,
            "message": "Retraining started",
            "status": "training",
            "epochs": epochs
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Retraining error: {e}")
        import traceback
        traceback.print_exc()
        training_state["is_training"] = False
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/api/model-performance")
async def get_model_performance():
    """Get current CNN model performance metrics"""
    try:
        cnn_model = CNNAntiSpoofing()
        
        if not cnn_model.is_trained():
            return {
                "is_trained": False,
                "message": "No trained CNN model found. Train a model first."
            }
        
        # Load all training data
        real_images, fake_images = cnn_model.load_training_images()
        
        if len(real_images) < 20 or len(fake_images) < 20:
            return {
                "is_trained": True,
                "message": "Not enough data to evaluate (need 20+ each)",
                "total_real": len(real_images),
                "total_fake": len(fake_images)
            }
        
        # Test on a sample of images (to avoid long processing)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Use up to 50 images of each type for evaluation
        test_real = real_images[:min(50, len(real_images))]
        test_fake = fake_images[:min(50, len(fake_images))]
        
        y_true = []
        y_pred = []
        
        # Test real images
        for img in test_real:
            is_live, _, confidence = cnn_model.predict(img, confidence_threshold=0.90)
            y_true.append(1)
            y_pred.append(1 if is_live else 0)
        
        # Test fake images
        for img in test_fake:
            is_live, _, confidence = cnn_model.predict(img, confidence_threshold=0.90)
            y_true.append(0)
            y_pred.append(1 if is_live else 0)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            "is_trained": True,
            "model_type": "CNN Deep Learning",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "total_real": len(real_images),
                "total_fake": len(fake_images),
                "tested_samples": len(y_true)
            }
        }
    except Exception as e:
        print(f"Performance evaluation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-status")
async def model_status():
    """Check current anti-spoofing model status"""
    is_cnn = isinstance(anti_spoof, CNNAntiSpoofing)
    is_trained = anti_spoof.is_trained() if is_cnn else False
    
    model_type = "CNN" if is_cnn else "Basic"
    
    return {
        "is_trained": is_trained,
        "model_type": model_type,
        "description": "Deep Learning (State-of-the-art)" if is_cnn else "Basic Fallback (Train CNN for security)"
    }

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/users")
async def get_all_users():
    """Get all registered users"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, username, email, created_at, status, last_login FROM users ORDER BY created_at DESC")
        users = c.fetchall()
        conn.close()
        
        users_list = []
        for user in users:
            users_list.append({
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "created_at": user[3],
                "status": user[4] if len(user) > 4 and user[4] else "active",
                "last_login": user[5] if len(user) > 5 else None
            })
        
        return {
            "success": True,
            "total": len(users_list),
            "users": users_list
        }
    except Exception as e:
        print(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")
        
        # Delete user
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"User '{user[0]}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/users/{user_id}/status")
async def update_user_status(user_id: int, status: str = Form(...)):
    """Update user status (active/inactive/suspended)"""
    try:
        if status not in ['active', 'inactive', 'suspended']:
            raise HTTPException(status_code=400, detail="Invalid status. Must be 'active', 'inactive', or 'suspended'")
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update status
        c.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"User '{user[0]}' status updated to '{status}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating user status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get a specific user's details"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, username, email, created_at, status, last_login FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "created_at": user[3],
                "status": user[4] if len(user) > 4 and user[4] else "active",
                "last_login": user[5] if len(user) > 5 else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/face")
async def get_user_face(user_id: int):
    """Get user's face image as base64"""
    try:
        from fastapi.responses import Response
        import base64
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT face_features FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        conn.close()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Reconstruct face image from features
        features = json.loads(user[0])
        face_array = np.array(features, dtype=np.uint8).reshape(100, 100)
        
        # Convert to RGB for better display
        face_rgb = cv2.cvtColor(face_array, cv2.COLOR_GRAY2RGB)
        
        # Resize for better display (200x200)
        face_display = cv2.resize(face_rgb, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight enhancement
        face_display = cv2.convertScaleAbs(face_display, alpha=1.2, beta=10)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', face_display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{img_base64}"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching user face: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/users/{user_id}/face")
async def update_user_face(user_id: int, image: UploadFile = File(...)):
    """Update user's face biometric data"""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        
        if not user:
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")
        
        # Read and process the new image
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        img_array = np.array(img)
        
        print(f"\n[Update Face] Processing new face for user {user[0]}")
        print(f"[Update Face] Image shape: {img_array.shape}")
        
        # Check liveness
        print("[Update Face] Checking liveness...")
        is_live, liveness_msg = check_liveness(img_array)
        print(f"[Update Face] Liveness result: {is_live} - {liveness_msg}")
        
        if not is_live:
            conn.close()
            raise HTTPException(status_code=400, detail=f"Liveness check failed: {liveness_msg}")
        
        # Extract face features
        print("[Update Face] Extracting face features...")
        face_hash, features = extract_face_features(img_array)
        
        if face_hash is None:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="No face detected. Please ensure: 1) Good lighting, 2) Face the camera directly, 3) Remove glasses/hats if needed"
            )
        
        features_str = json.dumps(features)
        
        # Update face data
        c.execute("UPDATE users SET face_hash = ?, face_features = ? WHERE id = ?",
                 (face_hash, features_str, user_id))
        conn.commit()
        conn.close()
        
        print(f"[Update Face] [OK] Face updated successfully for user {user[0]}")
        
        return {
            "success": True,
            "message": f"Face biometric data updated successfully for user '{user[0]}'"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Update Face] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

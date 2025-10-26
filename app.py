import streamlit as st
import cv2
import dlib
import numpy as np
import urllib.request
import os
import bz2
import shutil
import pickle
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from deepface import DeepFace
from fer import FER
import mediapipe as mp
from gtts import gTTS
import io

# =========================================================================
# 🔴 1. FULL SYSTEM SETUP, CACHING & IMPORTS
# =========================================================================

# --- Model & File Paths ---
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
MEDIAPIPE_TASK_PATH = "face_landmarker.task"
FACE_DB_PATH = "face_user_db_multi.pkl"
HAND_DB_PATH = "hand_user_db1.pkl"
CLASSIFIER_PATH = "gesture_classifier.pkl"
GESTURE_COMMANDS_PATH = "gesture_commands.pkl"

# --- Caching and Initialization ---
@st.cache_resource
def load_and_initialize_resources():
    st.info("Initializing models and loading databases...")

    # Download Models (Simplified for Streamlit deployment)
    if not os.path.exists(DLIB_LANDMARK_PATH):
        st.info("Downloading shape predictor (68 landmarks)...")
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        # Download and decompress
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, f"{DLIB_LANDMARK_PATH}.bz2")
        with bz2.open(f"{DLIB_LANDMARK_PATH}.bz2", "rb") as f_in:
            with open(DLIB_LANDMARK_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Note: Mediapipe task file is commented out as it's not strictly used in the current version of your code, but kept for completeness
    # if not os.path.exists(MEDIAPIPE_TASK_PATH):
    #     st.info("Downloading Mediapipe Face Landmarker task...")
    #     os.system(f"wget -O {MEDIAPIPE_TASK_PATH} https://storage.googleapis.com/mediapipe-assets/face_landmarker.task")
    
    # Initialize Detectors
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
    fer_detector = FER(mtcnn=False)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

    # Load Databases
    face_user_db = {}
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "rb") as f: face_user_db = pickle.load(f)

    hand_user_db = {}
    if os.path.exists(HAND_DB_PATH):
        with open(HAND_DB_PATH, "rb") as f: hand_user_db = pickle.load(f)

    # Load Classifier
    gesture_clf = None
    gesture_commands = {}
    try:
        with open(CLASSIFIER_PATH, "rb") as f: gesture_clf = pickle.load(f)
        with open(GESTURE_COMMANDS_PATH, "rb") as f: gesture_commands = pickle.load(f)
        st.success("✅ Classifier and commands loaded.")
    except Exception as e:
        st.warning(f"⚠️ Classifier/Commands not found. Run enrollment first to train. ({e})")

    st.success("✅ System initialization complete.")
    return dlib_detector, dlib_predictor, fer_detector, hands, face_user_db, hand_user_db, gesture_clf, gesture_commands

dlib_detector, dlib_predictor, fer_detector, hands, face_user_db, hand_user_db, gesture_clf, gesture_commands = load_and_initialize_resources()

# --- Session State Management ---
if 'face_user_db' not in st.session_state:
    st.session_state.face_user_db = face_user_db
if 'hand_user_db' not in st.session_state:
    st.session_state.hand_user_db = hand_user_db
if 'gesture_clf' not in st.session_state:
    st.session_state.gesture_clf = gesture_clf
if 'gesture_commands' not in st.session_state:
    st.session_state.gesture_commands = gesture_commands
if 'username' not in st.session_state:
    st.session_state.username = ""

# =========================================================================
# 🛠️ HELPER FUNCTIONS (Refactored for PIL/Bytes/Numpy)
# =========================================================================

# Helper to convert Streamlit's UploadedFile to a numpy array (OpenCV format)
def file_to_cv2(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return frame

# --- IRIS Feature Extraction (using LBP) ---
def extract_iris_dlib(gray, landmarks, eye_points):
    # ... (Keep this function as it is, it takes numpy arrays)
    xs = [landmarks.part(p).x for p in eye_points]
    ys = [landmarks.part(p).y for p in eye_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Ensure indices are valid
    h, w = gray.shape
    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    
    eye_img = gray[y_min:y_max, x_min:x_max]
    
    if eye_img.size == 0 or y_max <= y_min or x_max <= x_min: return None
    try:
        eye_img = cv2.resize(eye_img, (64,64))
    except cv2.error:
        return None
    return eye_img

def get_iris_features(frame):
    if frame is None: return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)
    feats = []
    
    for face in faces:
        landmarks = dlib_predictor(gray, face)
        for eye_points in [range(36,42), range(42,48)]:
            eye_img = extract_iris_dlib(gray, landmarks, eye_points)
            if eye_img is not None:
                lbp = local_binary_pattern(eye_img, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
                hist = hist.astype("float"); hist /= (hist.sum() + 1e-6)
                feats.append(hist.reshape(1, -1))
    return np.mean(feats, axis=0) if feats else None

# --- HAND Feature Extraction ---
def extract_hand_landmarks(frame):
    # ... (Keep this function as it is, it takes a numpy array)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks: return None
    lm = results.multi_hand_landmarks[0]
    points = np.array([(l.x, l.y) for l in lm.landmark])
    center = np.mean(points, axis=0)
    points -= center
    norm = np.linalg.norm(points)
    if norm > 0: points /= norm
    return points.flatten()

# --- Voice Output ---
def speak(text):
    if text:
        try:
            tts = gTTS(text, lang='en')
            # Use a BytesIO object to store the audio data in memory
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            st.audio(mp3_fp.getvalue(), format='audio/mp3')
        except Exception as e:
            st.error(f"Failed to generate voice output: {e}")

# =========================================================================
# 🎯 2. AUTHENTICATION HELPERS
# =========================================================================

def authenticate_face(frame, username, threshold=0.35):
    if username not in st.session_state.face_user_db: return False
    
    # Save frame temporarily to disk (DeepFace requires a path or a known format)
    # Using a named file for DeepFace to ensure it works reliably
    temp_path = f"temp_face_auth_{username}.jpg"
    cv2.imwrite(temp_path, frame)
    
    try:
        cap_emb = DeepFace.represent(temp_path, enforce_detection=True, model_name="Facenet512")[0]["embedding"]
    except Exception as e:
        st.error(f"Face detection/representation failed: {e}")
        return False
        
    best_similarity = -1
    for emb in st.session_state.face_user_db[username]:
        # Using a safer way to compute cosine similarity manually, as DeepFace's default metrics are distance, not similarity
        cos_sim = np.dot(emb, cap_emb) / (np.linalg.norm(emb) * np.linalg.norm(cap_emb))
        if cos_sim > best_similarity:
            best_similarity = cos_sim
            
    os.remove(temp_path) # Clean up
    st.info(f"Face Debug: Best Cosine Similarity: {best_similarity:.3f}, Threshold={threshold}")
    return best_similarity >= threshold

def authenticate_iris(frame, username, threshold=0.6):
    # This feature requires prior setup (not in your provided code), 
    # so we use a dummy enrolled feature as in your original code.
    iris_enrolled_feat = np.ones((1, 9)) * 0.111 
    current_feat = get_iris_features(frame)
    if current_feat is None:
        st.info("Iris Debug: Features NOT extracted (Dlib/LBP failed).")
        return False
    
    sim = cosine_similarity(iris_enrolled_feat, current_feat)[0][0]
    st.info(f"Iris Debug: Cosine Similarity: {sim:.3f}, Threshold={threshold}")
    return sim >= threshold

def authenticate_hand(frame, username, threshold=2.5):
    if username not in st.session_state.hand_user_db: return False
    emb = extract_hand_landmarks(frame)
    if emb is None:
        st.info("Hand Debug: Landmarks NOT detected in frame.")
        return False

    distances = [np.linalg.norm(emb - e) for e in st.session_state.hand_user_db[username]]
    best_dist = min(distances)
    
    st.info(f"Hand Debug: Best distance={best_dist:.3f}, Threshold={threshold}")
    return best_dist < threshold

# =========================================================================
# 🚀 3. MASTER FLOWS: AUTH & GESTURE-EMOTION
# =========================================================================

def master_multi_modal_authenticate(username, frame):
    # Save a temporary file for DeepFace, as it often expects a path
    temp_path = f"temp_auth_img_{username}.jpg"
    cv2.imwrite(temp_path, frame)

    auth_face = authenticate_face(frame, username)
    auth_iris = authenticate_iris(frame, username)
    auth_hand = authenticate_hand(frame, username)
    
    os.remove(temp_path) # Clean up

    auth_results = {"Face": auth_face, "Iris": auth_iris, "Hand": auth_hand}
    successful_matches = sum(auth_results.values())
    
    st.subheader("--- Multi-Modal Authentication Results ---")
    st.write(f"**Face Match:** {auth_face}")
    st.write(f"**Iris Match:** {auth_iris}")
    st.write(f"**Hand Match:** {auth_hand}")
    
    if successful_matches >= 2:
        status = f"✅ AUTHENTICATION PASSED! ({successful_matches}/3 modalities matched)"
        return True, status
    else:
        status = f"❌ AUTHENTICATION FAILED. ({successful_matches}/3 modalities matched)"
        return False, status

def master_gesture_emotion_voice(frame):
    if st.session_state.gesture_clf is None:
        st.error("❌ Cannot run. Gesture classifier not trained. Run enrollment first.")
        return

    st.subheader("--- Running Gesture-Emotion Command ---")

    # 1. Detect Gesture
    detected_gesture = None
    emb = extract_hand_landmarks(frame)
    if emb is not None:
        pred = st.session_state.gesture_clf.predict([emb])[0]
        if pred in st.session_state.gesture_commands:
            detected_gesture = pred
    
    if detected_gesture is None:
        st.warning("❌ No recognizable gesture detected.")
        return

    # 2. Detect Emotion (Single frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = fer_detector.detect_emotions(frame_rgb)
    dominant_emotion = "neutral"
    
    if results:
        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        st.markdown("\n**Emotion Debug (Confidence):**")
        emotion_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Confidence'])
        st.dataframe(emotion_df.sort_values(by='Confidence', ascending=False).set_index('Emotion'), use_container_width=True)
        
        if confidence < 0.3:
            dominant_emotion = "neutral"
            st.warning(f"⚠️ Confidence for {dominant_emotion} is low (<0.3), defaulting to neutral.")

    st.success(f"🎯 Detected Gesture: {detected_gesture}")
    st.success(f"🎭 Final Emotion: {dominant_emotion}")

    # 3. Command Mapping & Voice Output
    base_command = st.session_state.gesture_commands.get(detected_gesture, "I don't know the command.")

    final_text = base_command
    if detected_gesture == "FIVE" and dominant_emotion == "fear":
        final_text = "EMERGENCY! I need HELP immediately!"
    elif detected_gesture == "FIST" and dominant_emotion == "angry":
        final_text = "STOP! I am highly displeased with this."
    elif dominant_emotion == "happy":
        final_text = "That's wonderful! " + base_command
    
    st.markdown(f"**🗣️ Final Voice Command:** *'{final_text}'*")
    speak(final_text)

# =========================================================================
# 4. USER INTERFACE (STREAMLIT) & HANDLERS
# =========================================================================

def register_face(name, frames):
    embeddings = []
    
    for i, frame in enumerate(frames):
        temp_path = f"temp_face_enroll_{name}_{i}.jpg"
        cv2.imwrite(temp_path, frame)
        try:
            emb = DeepFace.represent(temp_path, enforce_detection=True, model_name="Facenet512")[0]["embedding"]
            embeddings.append(emb)
        except Exception as e:
            st.error(f"Face registration failed for sample {i+1}: {e}")
        os.remove(temp_path) # Clean up

    if embeddings:
        st.session_state.face_user_db[name] = embeddings
        with open(FACE_DB_PATH, "wb") as f: pickle.dump(st.session_state.face_user_db, f)
        st.success(f"✅ Face DB saved for '{name}'.")
    else:
        st.error(f"❌ No valid face embeddings captured for '{name}'.")

def register_hand(name, frames):
    embeddings = []
    for frame in frames:
        emb = extract_hand_landmarks(frame)
        if emb is not None: embeddings.append(emb)
    if embeddings:
        st.session_state.hand_user_db[name] = embeddings
        with open(HAND_DB_PATH, "wb") as f: pickle.dump(st.session_state.hand_user_db, f)
        st.success(f"✅ Hand DB saved for '{name}'.")
    else:
        st.error(f"❌ No valid hand landmarks captured for '{name}'.")

# --- Streamlit UI ---

st.title("🛡️ Multi-Modal Biometric & Command System")
st.caption("A computer vision system for authentication and gesture/emotion commands.")

st.markdown("---")

st.header("1. Enrollment/Training")
st.session_state.username = st.text_input("Enter a User Name for Enrollment/Auth:", value=st.session_state.username)

enroll_tab, train_tab = st.tabs(["Enroll Biometrics", "Train Gestures"])

with enroll_tab:
    st.markdown("##### Capture Enrollment Samples (Face, Iris, and Open Hand)")
    enroll_img = st.camera_input("Take 2 photos, showing your face and an open hand clearly.", key="enroll_camera")

    if st.button("Start Enrollment Process", disabled=(not st.session_state.username or not enroll_img)):
        if st.session_state.username in st.session_state.face_user_db:
             st.warning(f"User '{st.session_state.username}' already exists. Overwriting data.")
        
        # --- Streamlit Enrollment Flow ---
        st.info("Enrollment requires multiple images. Please take 2 photos now.")
        
        # In a real Streamlit app, you'd capture multiple photos sequentially using a loop 
        # with button clicks, but for simplicity here we just use the one image captured.
        # A more robust app would prompt for 2-3 images.
        
        frame = file_to_cv2(enroll_img)
        
        with st.spinner("Processing enrollment image..."):
            # Check for hand before proceeding (enforcement)
            if extract_hand_landmarks(frame) is None:
                st.error("❌ Hand NOT detected in the image. Please retake the photo with your hand visible.")
            else:
                st.success("✅ Hand detected successfully. Proceeding with registration.")
                
                # Use the single captured frame as both face/iris and hand samples
                register_face(st.session_state.username, [frame, frame]) 
                register_hand(st.session_state.username, [frame, frame]) 
                st.success("🎉 Biometric Enrollment Complete.")
                

with train_tab:
    st.markdown("##### Train Gesture Commands (FIVE, FIST, OK)")
    st.info("This step requires taking *6 separate photos* (2 for each gesture).")
    
    # We will use st.session_state to manage the training data collection
    if 'gesture_data' not in st.session_state:
        st.session_state.gesture_data = []
    
    if st.button("Clear Gesture Data"):
        st.session_state.gesture_data = []
        st.experimental_rerun()

    current_gesture = st.selectbox("Select Gesture to Capture:", ["FIVE", "FIST", "OK"])
    gesture_command_img = st.camera_input(f"Take a photo of the '{current_gesture}' gesture.", key="gesture_camera")
    
    if st.button(f"Add 1 Sample for {current_gesture}", disabled=(not gesture_command_img)):
        frame = file_to_cv2(gesture_command_img)
        emb = extract_hand_landmarks(frame)
        if emb is not None:
            st.session_state.gesture_data.append([current_gesture] + list(emb))
            st.success(f"Sample added for {current_gesture}. Total samples: {len(st.session_state.gesture_data)}")
        else:
            st.error("❌ Hand not detected in the gesture sample. Please retake.")
            
    st.info(f"Currently collected samples: **{len(st.session_state.gesture_data)}** (Need at least 6 to train 3 gestures)")
    
    if st.button("Train Gesture Classifier", disabled=(len(st.session_state.gesture_data) < 6)):
        with st.spinner("Training Random Forest Classifier..."):
            df = pd.DataFrame(st.session_state.gesture_data)
            X = df.iloc[:,1:].values; y = df.iloc[:,0].values
            
            # Train and save
            st.session_state.gesture_clf = RandomForestClassifier(n_estimators=100).fit(X, y)
            with open(CLASSIFIER_PATH, "wb") as f: pickle.dump(st.session_state.gesture_clf, f)
            
            # Save default commands
            default_phrases = {"FIVE": "Hello", "FIST": "Goodbye", "OK": "Affirmative"}
            st.session_state.gesture_commands = default_phrases
            with open(GESTURE_COMMANDS_PATH, "wb") as f: pickle.dump(st.session_state.gesture_commands, f)
            
            st.success("✅ Gesture Classifier trained and saved.")
            st.balloons()
            st.session_state.gesture_data = [] # Clear data after training

st.markdown("---")

st.header("2. Run Actions")

action_col, camera_col = st.columns([1, 1])

with camera_col:
    st.markdown("##### Live Camera Feed")
    action_img = st.camera_input("Capture image for Authentication or Command", key="action_camera")
    
with action_col:
    if st.button("Run Multi-Modal Auth", disabled=(not action_img or not st.session_state.username)):
        if st.session_state.username not in st.session_state.face_user_db:
             st.error(f"User '{st.session_state.username}' not enrolled. Please enroll first.")
        else:
            with st.spinner("Authenticating..."):
                frame = file_to_cv2(action_img)
                st.image(frame, channels="BGR", caption="Authentication Frame", use_column_width=True)
                success, message = master_multi_modal_authenticate(st.session_state.username, frame)
                st.write(message)
                if success:
                    speak("Authentication successful!")

    if st.button("Run Gesture-Emotion Voice Command", disabled=(not action_img or st.session_state.gesture_clf is None)):
        if st.session_state.gesture_clf is None:
            st.error("Gesture classifier is not trained. Please train it first.")
        else:
            with st.spinner("Analyzing Gesture and Emotion..."):
                frame = file_to_cv2(action_img)
                st.image(frame, channels="BGR", caption="Command Frame", use_column_width=True)
                master_gesture_emotion_voice(frame)

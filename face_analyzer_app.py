"""
Face Feature Analyzer & Virtual Effects App
A creative facial recognition application using OpenCV
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Configure page
st.set_page_config(
    page_title="Face Feature Analyzer",
    page_icon="👤",
    layout="wide"
)

# Title and description
st.title("👤 Face Feature Analyzer & Virtual Effects")
st.markdown("""
Upload an image and discover:
- **Face Detection** - Find all faces in the image
- **Facial Landmarks** - Map 68 facial feature points
- **Feature Analysis** - Analyze face shape, symmetry, and more
- **Virtual Effects** - Apply fun filters and privacy blurring
""")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.05)
feature_mode = st.sidebar.selectbox(
    "Choose Feature",
    ["Face Detection", "Facial Landmarks", "Feature Analysis", "Privacy Blur", "Virtual Glasses"]
)

# Load models (cached for performance)
@st.cache_resource
def load_face_detector():
    """Load the face detection model"""
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_path = "deploy.prototxt"
    try:
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        return net
    except:
        st.error("⚠️ Model files not found. Please ensure deploy.prototxt and .caffemodel are in the same directory.")
        return None

@st.cache_resource
def load_landmark_detector():
    """Load the facial landmark detector"""
    import urllib.request
    import os
    
    model_file = "lbfmodel.yaml"
    
    # Download model if it doesn't exist
    if not os.path.exists(model_file):
        try:
            st.info("📥 Downloading landmark model (52MB, one-time only)... Please wait ~30 seconds.")
            url = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
            urllib.request.urlretrieve(url, model_file)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    
    try:
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(model_file)
        return landmark_detector
    except Exception as e:
        st.warning(f"⚠️ Landmark model could not be loaded: {e}")
        return None

# Helper functions
def detect_faces(image, net, threshold=0.7):
    """Detect faces in the image"""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2, y2))
    
    return faces

def draw_face_boxes(image, faces):
    """Draw bounding boxes around detected faces"""
    img_out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(faces):
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_out, f"Face {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img_out

def detect_landmarks(image, faces, detector):
    """Detect facial landmarks"""
    if detector is None:
        return None
    
    retval, landmarks = detector.fit(image, faces)
    return landmarks

def draw_landmarks(image, landmarks):
    """Draw facial landmarks on the image"""
    img_out = image.copy()
    if landmarks:
        for landmark_set in landmarks:
            for point in landmark_set[0]:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img_out, (x, y), 2, (0, 255, 255), -1)
    return img_out

def analyze_face_features(landmarks):
    """Analyze facial features from landmarks"""
    if not landmarks or len(landmarks) == 0:
        return {}
    
    points = landmarks[0][0]
    
    # Calculate face measurements
    # Jaw points: 0-16
    jaw_width = np.linalg.norm(points[0] - points[16])
    
    # Eye points: left 36-41, right 42-47
    left_eye_width = np.linalg.norm(points[36] - points[39])
    right_eye_width = np.linalg.norm(points[42] - points[45])
    eye_distance = np.linalg.norm(points[39] - points[42])
    
    # Nose points: 27-35
    nose_length = np.linalg.norm(points[27] - points[33])
    
    # Mouth points: 48-67
    mouth_width = np.linalg.norm(points[48] - points[54])
    
    # Face shape analysis
    face_height = np.linalg.norm(points[8] - points[27])
    face_ratio = face_height / jaw_width if jaw_width > 0 else 0
    
    if face_ratio > 1.3:
        face_shape = "Oval"
    elif face_ratio < 1.1:
        face_shape = "Round"
    else:
        face_shape = "Balanced"
    
    return {
        "Face Shape": face_shape,
        "Jaw Width": f"{jaw_width:.1f}px",
        "Eye Distance": f"{eye_distance:.1f}px",
        "Nose Length": f"{nose_length:.1f}px",
        "Mouth Width": f"{mouth_width:.1f}px",
        "Symmetry Score": f"{min(left_eye_width/right_eye_width, right_eye_width/left_eye_width)*100:.1f}%"
    }

def apply_blur(image, faces, blur_type="elliptical"):
    """Apply privacy blur to detected faces"""
    img_out = image.copy()
    
    for (x1, y1, x2, y2) in faces:
        face_roi = img_out[y1:y2, x1:x2]
        
        if blur_type == "pixelated":
            # Pixelated blur
            h, w = face_roi.shape[:2]
            temp = cv2.resize(face_roi, (w//15, h//15), interpolation=cv2.INTER_LINEAR)
            face_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # Gaussian blur
            ksize = ((x2-x1)//3, (y2-y1)//3)
            ksize = (ksize[0]|1, ksize[1]|1)  # Make odd
            face_roi = cv2.GaussianBlur(face_roi, ksize, 0)
        
        img_out[y1:y2, x1:x2] = face_roi
    
    return img_out

def add_virtual_glasses(image, landmarks):
    """Add virtual glasses effect"""
    if not landmarks or len(landmarks) == 0:
        return image
    
    img_out = image.copy()
    points = landmarks[0][0].astype(int)
    
    # Get eye positions (left: 36-41, right: 42-47)
    left_eye_center = np.mean(points[36:42], axis=0).astype(int)
    right_eye_center = np.mean(points[42:48], axis=0).astype(int)
    
    # Calculate glasses parameters
    eye_distance = int(np.linalg.norm(left_eye_center - right_eye_center))
    glasses_width = int(eye_distance * 2.5)
    glasses_height = int(glasses_width * 0.4)
    
    # Center point between eyes
    center = ((left_eye_center + right_eye_center) // 2).astype(int)
    
    # Draw glasses frame
    # Left lens
    cv2.ellipse(img_out, 
               tuple(left_eye_center), 
               (eye_distance//3, glasses_height//2),
               0, 0, 360, (0, 0, 0), 3)
    
    # Right lens
    cv2.ellipse(img_out, 
               tuple(right_eye_center), 
               (eye_distance//3, glasses_height//2),
               0, 0, 360, (0, 0, 0), 3)
    
    # Bridge
    cv2.line(img_out, tuple(left_eye_center + [eye_distance//3, 0]), 
            tuple(right_eye_center - [eye_distance//3, 0]), (0, 0, 0), 3)
    
    # Temples
    temple_start_l = tuple(left_eye_center - [eye_distance//3, 0])
    temple_start_r = tuple(right_eye_center + [eye_distance//3, 0])
    cv2.line(img_out, temple_start_l, tuple(points[0]), (0, 0, 0), 3)
    cv2.line(img_out, temple_start_r, tuple(points[16]), (0, 0, 0), 3)
    
    return img_out

# Main app
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Load models
    net = load_face_detector()
    landmark_detector = load_landmark_detector()
    
    if net is not None:
        # Detect faces
        faces = detect_faces(img_cv, net, detection_confidence)
        
        # Display results based on selected feature
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader(f"Result: {feature_mode}")
            
            if len(faces) == 0:
                st.warning("⚠️ No faces detected. Try lowering the detection confidence.")
                st.image(image, use_container_width=True)
            else:
                st.success(f"✅ Detected {len(faces)} face(s)")
                
                # Process based on feature mode
                if feature_mode == "Face Detection":
                    result_img = draw_face_boxes(img_cv, faces)
                    
                elif feature_mode == "Facial Landmarks":
                    if landmark_detector:
                        landmarks = detect_landmarks(img_cv, faces, landmark_detector)
                        result_img = draw_landmarks(img_cv, landmarks)
                        result_img = draw_face_boxes(result_img, faces)
                    else:
                        st.error("Landmark detector not available")
                        result_img = img_cv
                
                elif feature_mode == "Feature Analysis":
                    if landmark_detector:
                        landmarks = detect_landmarks(img_cv, faces, landmark_detector)
                        result_img = draw_landmarks(img_cv, landmarks)
                        
                        # Show analysis
                        analysis = analyze_face_features(landmarks)
                        st.markdown("### 📊 Face Analysis")
                        for key, value in analysis.items():
                            st.metric(key, value)
                    else:
                        st.error("Landmark detector not available")
                        result_img = img_cv
                
                elif feature_mode == "Privacy Blur":
                    blur_style = st.radio("Blur Style", ["Gaussian", "Pixelated"], horizontal=True)
                    blur_type = "pixelated" if blur_style == "Pixelated" else "elliptical"
                    result_img = apply_blur(img_cv, faces, blur_type)
                
                elif feature_mode == "Virtual Glasses":
                    if landmark_detector:
                        landmarks = detect_landmarks(img_cv, faces, landmark_detector)
                        result_img = add_virtual_glasses(img_cv, landmarks)
                    else:
                        st.error("Landmark detector not available")
                        result_img = img_cv
                
                # Convert back to RGB for display
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
                
                # Download button
                result_pil = Image.fromarray(result_rgb)
                buf = io.BytesIO()
                result_pil.save(buf, format='PNG')
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name=f"face_analysis_{feature_mode.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using OpenCV and Streamlit | Face Detection + Facial Landmarks + Feature Analysis</p>
</div>
""", unsafe_allow_html=True)

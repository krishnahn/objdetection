import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import io

# Set page configuration
st.set_page_config(
    page_title="Object recognition",
    page_icon="🔍",
    layout="wide",
)

# App title and description
st.title("Object recognition System")
st.subheader("Upload two objects, train the model, and detect them with your webcam")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'features' not in st.session_state:
    st.session_state.features = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'object1_name' not in st.session_state:
    st.session_state.object1_name = "Object 1"
if 'object2_name' not in st.session_state:
    st.session_state.object2_name = "Object 2"

# Create directory for storing temporary images
if not os.path.exists('temp_images'):
    os.makedirs('temp_images')

def preprocess_image(image):
    """
    Preprocess image for consistent feature extraction
    """
    try:
        # Convert to RGB if needed
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize for consistency
        image = cv2.resize(image, (224, 224))
        
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        return image
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def extract_features(image):
    """
    Enhanced feature extraction with additional features and error handling
    """
    try:
        # Preprocess image
        img = preprocess_image(image)
        if img is None:
            return None
        
        # Ensure image is in uint8 format
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Convert to grayscale for SIFT
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Histogram features
        hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist_color = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
        
        # 2. SIFT features
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                sift_features = np.mean(descriptors, axis=0)
            else:
                sift_features = np.zeros(128)
        except Exception as e:
            st.warning(f"SIFT feature extraction failed: {str(e)}")
            sift_features = np.zeros(128)
        
        # 3. Basic shape features
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            shape_features = [area, perimeter]
        else:
            shape_features = [0, 0]
        
        # 4. Color features
        color_features = []
        for channel in cv2.split(img):
            color_features.extend([
                np.mean(channel),
                np.std(channel),
                np.mean(np.abs(channel - np.mean(channel)))
            ])
        
        # Combine all features
        features = np.concatenate([
            hist_gray.flatten(),
            np.concatenate([h.flatten() for h in hist_color]),
            sift_features,
            shape_features,
            color_features
        ])
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def train_model(features, labels):
    """
    Enhanced model training with cross-validation
    """
    try:
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train logistic regression model with increased max_iter and balanced class weights
        model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            C=1.0,
            solver='lbfgs'
        )
        model.fit(features_scaled, labels)
        
        return model, scaler
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None

def predict_object(model, scaler, image, confidence_threshold=0.65):
    """
    Enhanced prediction with confidence threshold
    """
    try:
        features = extract_features(image)
        if features is None:
            return None, 0
        
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        
        # Only return prediction if confidence is above threshold
        if confidence >= confidence_threshold:
            return prediction, confidence
        else:
            return None, confidence
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, 0

# Training section
st.header("Step 1: Train the Model")
train_col1, train_col2 = st.columns(2)

with train_col1:
    st.subheader("Upload Object 1")
    st.session_state.object1_name = st.text_input("Name for Object 1", "Object 1")
    object1_image = st.file_uploader("Upload image of Object 1", type=["jpg", "jpeg", "png"])
    if object1_image:
        st.image(object1_image, caption=st.session_state.object1_name, width=300)

with train_col2:
    st.subheader("Upload Object 2")
    st.session_state.object2_name = st.text_input("Name for Object 2", "Object 2")
    object2_image = st.file_uploader("Upload image of Object 2", type=["jpg", "jpeg", "png"])
    if object2_image:
        st.image(object2_image, caption=st.session_state.object2_name, width=300)

# Train model button
if object1_image and object2_image:
    if st.button("Train Model", key="train_button"):
        with st.spinner("Training model... This might take a few seconds."):
            try:
                # Process first object
                img1 = Image.open(object1_image).convert('RGB')
                img1_array = np.array(img1)
                
                # Create augmented versions of object 1
                features = []
                labels = []
                
                # Original image
                features1 = extract_features(img1_array)
                if features1 is not None:
                    features.append(features1)
                    labels.append(0)
                
                    # Augmented images for object 1
                    for i in range(9):
                        # Create augmented image
                        img_mod = img1_array.copy()
                        
                        # Random rotation
                        angle = np.random.uniform(-15, 15)
                        rows, cols = img_mod.shape[:2]
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        img_mod = cv2.warpAffine(img_mod, M, (cols, rows))
                        
                        # Random brightness and contrast
                        brightness = np.random.uniform(0.8, 1.2)
                        contrast = np.random.uniform(0.8, 1.2)
                        img_mod = np.clip(img_mod * contrast + brightness, 0, 255).astype(np.uint8)
                        
                        features_aug = extract_features(img_mod)
                        if features_aug is not None:
                            features.append(features_aug)
                            labels.append(0)
                
                # Process second object
                img2 = Image.open(object2_image).convert('RGB')
                img2_array = np.array(img2)
                
                # Original image
                features2 = extract_features(img2_array)
                if features2 is not None:
                    features.append(features2)
                    labels.append(1)
                
                    # Augmented images for object 2
                    for i in range(9):
                        # Create augmented image
                        img_mod = img2_array.copy()
                        
                        # Random rotation
                        angle = np.random.uniform(-15, 15)
                        rows, cols = img_mod.shape[:2]
                        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                        img_mod = cv2.warpAffine(img_mod, M, (cols, rows))
                        
                        # Random brightness and contrast
                        brightness = np.random.uniform(0.8, 1.2)
                        contrast = np.random.uniform(0.8, 1.2)
                        img_mod = np.clip(img_mod * contrast + brightness, 0, 255).astype(np.uint8)
                        
                        features_aug = extract_features(img_mod)
                        if features_aug is not None:
                            features.append(features_aug)
                            labels.append(1)
                
                if len(features) > 0:
                    # Train model
                    features_array = np.array(features)
                    labels_array = np.array(labels)
                    
                    model, scaler = train_model(features_array, labels_array)
                    
                    if model is not None and scaler is not None:
                        # Save to session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.trained = True
                        st.session_state.features = features_array
                        st.session_state.labels = labels_array
                        
                        st.success("✅ Model trained successfully! Now you can test it.")
                    else:
                        st.error("Failed to train model. Please try again with different images.")
                else:
                    st.error("Failed to extract features from images. Please try different images.")
            
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

# Testing section
if st.session_state.trained:
    st.header("Step 2: Test the Model")
    
    test_method = st.radio("Choose test method:", 
                          ["Upload an image", "Use webcam"], 
                          horizontal=True)
    
    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        help="Minimum confidence level required for prediction"
    )
    
    if test_method == "Use webcam":
        camera_image = st.camera_input("Take a picture of one of your objects")
        
        if camera_image:
            # Show the camera image
            st.image(camera_image, caption="Captured Image", width=300)
            
            # Process the image
            image = Image.open(camera_image)
            prediction, confidence = predict_object(
                st.session_state.model,
                st.session_state.scaler,
                image,
                confidence_threshold
            )
            
            # Display result
            if prediction is not None:
                result_text = st.session_state.object1_name if prediction == 0 else st.session_state.object2_name
                st.success(f"Prediction: {result_text}")
                st.info(f"Confidence: {confidence*100:.2f}%")
            else:
                st.warning(f"Confidence too low ({confidence*100:.2f}%) to make a prediction. Please try again.")
    
    else:  # Upload an image
        test_image = st.file_uploader("Upload an image to test", type=["jpg", "jpeg", "png"], key="test_uploader")
        
        if test_image:
            # Show the uploaded image
            st.image(test_image, caption="Test Image", width=300)
            
            # Process the image
            image = Image.open(test_image)
            prediction, confidence = predict_object(
                st.session_state.model,
                st.session_state.scaler,
                image,
                confidence_threshold
            )
            
            # Display result
            if prediction is not None:
                result_text = st.session_state.object1_name if prediction == 0 else st.session_state.object2_name
                st.success(f"Prediction: {result_text}")
                st.info(f"Confidence: {confidence*100:.2f}%")
            else:
                st.warning(f"Confidence too low ({confidence*100:.2f}%) to make a prediction. Please try again.")

# Add debug information in expander
with st.expander("Debug Information"):
    st.write("Model Status:", "Trained" if st.session_state.trained else "Not trained")
    if st.session_state.trained:
        st.write("Number of training samples:", len(st.session_state.labels))
        st.write("Class distribution:", np.bincount(st.session_state.labels))
        st.write("Feature vector size:", st.session_state.features.shape[1])


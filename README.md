# Object Recognition System using Streamlit

A lightweight and interactive object recognition system built with **Streamlit**, allowing users to:

- Upload **two images** representing different objects.
- Train a **custom classifier on-the-fly** using those images.
- Classify live webcam input or uploaded images into **Object 1** or **Object 2**.

---

##  Features

- Upload one image for each object class.
- On-the-fly model training using Logistic Regression.
- Real-time image prediction using webcam or uploaded image.
- Augments data with rotation, brightness, and contrast variations.
- Uses color, shape, texture, and SIFT-based features.
- Adjustable confidence threshold for predictions.

---

## requirements

streamlit
opencv-python
numpy
Pillow
scikit-learn


## Project Structure
.
├── app.py              # Main Streamlit app (your uploaded code)
├── README.md           # This file
├── requirements.txt    # List of dependencies (optional)
└── temp_images/        # Temporary folder for images (auto-created at runtime)



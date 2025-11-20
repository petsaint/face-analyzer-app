# 👤 Face Feature Analyzer & Virtual Effects

A creative facial recognition web application that detects faces, analyzes facial features, and applies fun virtual effects.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## 🌟 Features

- **Face Detection**: Detect multiple faces with adjustable confidence
- **Facial Landmarks**: Detect 68 facial feature points with precision
- **Feature Analysis**: Analyze face shape, symmetry, and proportions
- **Privacy Blur**: Apply Gaussian or pixelated blur for privacy protection
- **Virtual Glasses**: Add fun virtual glasses filter using landmark detection

## 🚀 Live Demo

**Deployed App:** [Your Streamlit URL here]

## 📸 Screenshots

[Add screenshots of your app in action]

## 🛠️ Technologies Used

- **OpenCV**: Face detection and image processing
- **Streamlit**: Interactive web interface
- **NumPy**: Numerical computations
- **PIL/Pillow**: Image handling
- **Caffe DNN Model**: SSD face detection
- **LBF Model**: Facial landmark detection

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/face-analyzer-app.git
cd face-analyzer-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the model files:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- `lbfmodel.yaml`

4. Run the app:
```bash
streamlit run face_analyzer_app.py
```

5. Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
face-analyzer-app/
│
├── face_analyzer_app.py          # Main Streamlit application
├── requirements.txt               # Python dependencies
├── deploy.prototxt               # Caffe model architecture
├── res10_300x300_ssd_iter_140000_fp16.caffemodel  # Face detection weights
├── lbfmodel.yaml                 # Facial landmark model
└── README.md                     # This file
```

## 🎯 How to Use

1. **Upload an Image**: Click "Upload an image" and select a JPG or PNG file
2. **Adjust Settings**: Use the sidebar to adjust detection confidence
3. **Choose Feature**: Select from 5 different modes:
   - Face Detection
   - Facial Landmarks
   - Feature Analysis
   - Privacy Blur
   - Virtual Glasses
4. **Download Results**: Click the download button to save your processed image

## 🧠 Technical Details

### Face Detection
Uses OpenCV's DNN module with a pre-trained Caffe SSD (Single Shot Detector) model:
- Input: 300x300 image blob
- Output: Face bounding boxes with confidence scores
- Threshold: Adjustable (default 0.7)

### Facial Landmarks
Uses Local Binary Features (LBF) algorithm:
- Detects 68 facial landmarks
- Points include: jaw, eyebrows, eyes, nose, mouth
- Enables precise facial feature location

### Feature Analysis
Mathematical analysis of detected landmarks:
- Face shape classification (Oval/Round/Balanced)
- Facial proportions measurement
- Symmetry score calculation

## 🎨 Features in Detail

### 1. Face Detection Mode
- Draws green bounding boxes around detected faces
- Labels each face with a number
- Works with multiple faces in one image

### 2. Facial Landmarks Mode
- Maps 68 precise points on facial features
- Yellow dots show landmark positions
- Includes face bounding boxes

### 3. Feature Analysis Mode
- Calculates facial measurements
- Determines face shape
- Shows symmetry percentage
- Displays metrics in organized cards

### 4. Privacy Blur Mode
- **Gaussian Blur**: Smooth, professional blur
- **Pixelated Blur**: Retro pixelated effect
- Automatically applies to all detected faces

### 5. Virtual Glasses Mode
- Intelligently places glasses using eye landmarks
- Draws realistic glasses frames
- Adjusts size based on face proportions

## 🔧 Configuration

Adjust these parameters in the sidebar:
- **Detection Confidence**: 0.5 to 1.0 (default: 0.7)
- **Feature Mode**: 5 different modes available
- **Blur Style**: Gaussian or Pixelated (in Privacy mode)

## 📊 Model Information

### Face Detection Model
- **Architecture**: SSD (Single Shot Detector)
- **Framework**: Caffe
- **Input Size**: 300x300
- **Output**: Bounding boxes + confidence scores

### Landmark Detection Model  
- **Algorithm**: LBF (Local Binary Features)
- **Points**: 68 facial landmarks
- **Format**: YAML model file

## 🤝 Contributing

This is a student project for educational purposes. Feel free to fork and experiment!

## 📄 License

This project is created for educational purposes as part of a Computer Vision course assignment.

## 👨‍💻 Author

**[Your Name]**
- Course: Computer Vision
- Assignment: Facial Recognition Application
- Date: [Current Date]

## 🙏 Acknowledgments

- Course materials from [Your Course Name]
- OpenCV documentation and community
- Streamlit for the amazing framework
- Pre-trained models from OpenCV Model Zoo

## 📞 Support

For issues or questions:
1. Check the [Deployment Guide](DEPLOYMENT_GUIDE.md)
2. Review Streamlit logs for errors
3. Ensure all model files are present

---

**Built with ❤️ using OpenCV and Streamlit**

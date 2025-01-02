Here's a detailed README file for your **Emotion Detector Live** project:

---

# Emotion Detector Live 🎭

**Emotion Detector Live** is a real-time emotion recognition application powered by a fine-tuned Vision Transformer (ViT) model. This project uses live video input to detect and classify human emotions into seven categories: **angry, disgust, fear, happy, neutral, sad, and surprise**. 

With a **97% final training accuracy**, this model achieves impressive performance, making it a perfect tool for practical applications in AI-based emotion analysis.

---

## Key Features 🚀

- **Real-Time Emotion Detection**: Processes video feed live to identify emotions.
- **Trigger Button**: Detection starts and stops with the "P" key.
- **User-Friendly UI**: A green box surrounds the face, with the detected emotion displayed outside the box.
- **Accurate Predictions**: Achieves high precision, recall, and F1-score during validation.
- **Pre-Trained Vision Transformer**: Fine-tuned for high performance on a custom dataset.

---

## Model Training Details 📊

- **Model Architecture**: Vision Transformer (ViT) fine-tuned on a custom emotion dataset.
- **Dataset Size**: 35,000 images divided into training and testing datasets.
- **Final Training Accuracy**: **97%**
- **Validation Metrics**:  
  - **Precision**: *add value here*  
  - **Recall**: *add value here*  
  - **F1-Score**: *add value here*

### Model Performance

| Metric        | Value  |
|---------------|--------|
| Training Accuracy | 97.75%   |
| Validation Accuracy | 68.65% |
| Validation Precision     | 68.34% |
| Validation F1-Score      | 68.36% |

![Figure_1](https://github.com/user-attachments/assets/929e37ab-db2b-4f37-b861-4c111047e784)

---

## Setup and Installation 🛠️

### Prerequisites
1. Python 3.8+
2. `torch` and `torchvision`
3. `transformers`
4. `opencv-python`
5. GPU (Recommended for faster performance)

### Steps to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Emotion-Detector-Live.git
   cd Emotion-Detector-Live
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the `final_model.pth` and `vit_emotion_model.pth` files in the root directory.
4. Run the program:
   ```bash
   python main.py
   ```
5. Follow the on-screen instructions:
   - Press **P** to start/stop the emotion detection.
   - A green box will appear around the face, with the detected emotion displayed outside.

---

## Usage Instructions 🎥

1. Launch the program to activate your webcam.
2. Ensure your face is visible to the camera.
3. Press **P** to toggle emotion detection.
4. The green box will appear around your face, and the detected emotion will be displayed.
5. Press **P** again to stop the detection.

---

## Results 📸

### Screenshots:
*(Insert a screenshot of the final app interface here)*

### Example Images:
*(Include images of you using the emotion detector here)*

---

## Limitations ⚠️

- Works best in well-lit environments.
- Performance may vary with extreme facial expressions or occlusions.

---

## Future Enhancements 🌟

1. Extend to multi-face emotion detection.
2. Add support for additional emotions.
3. Integrate audio or text feedback based on detected emotions.
4. Optimize the model for mobile and embedded devices.

---

## Acknowledgments 🙏

This project is inspired by cutting-edge advancements in computer vision and transformers. Special thanks to the creators of the Vision Transformer and Hugging Face libraries.

---

## Contact 📬

For queries or suggestions, feel free to contact me:  
- **Email**: abhijithpv32@gmail.com  
- **GitHub**: [Abhijith p v](https://github.com/ab-hi-ji-th)

---

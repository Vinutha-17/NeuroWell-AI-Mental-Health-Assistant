# 🧠 NeuroWell – AI Mental Health Assistant

**NeuroWell** is an AI-powered mental wellness assistant that combines **emotion detection from face images** with a **context-aware chatbot** to provide support, motivation, and mental health tips. Designed for anyone looking for a quick emotional check-in and encouraging conversation.

---

## 🌟 Features

- **📷 Emotion Detection**
  - Upload your face image to detect emotions using a deep learning ONNX model.
  - Detects: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`.

- **💬 Mental Health Chatbot**
  - Intelligent, empathetic, and memory-aware conversational assistant.
  - Recognizes user mood and offers motivational quotes, tips, or calming suggestions.

- **🌐 Internet-Enabled**
  - Gradio interface runs in the browser but requires internet access to function properly.

- **🖥️ Interactive UI**
  - Simple and intuitive layout using [Gradio](https://www.gradio.app).
  - Two tabs: `Emotion Detector` and `Chat Assistant`.

---

## 🗂 Project Structure

NeuroWell/
├── models/
│ └── emotion_model.onnx # ONNX emotion classification model
├── gradio_emotion_app1.py # Main app script
├── requirements.txt # Python dependency list
└── README.md # This file

---

## 🔧 Installation

### 📌 Requirements

- Python 3.8+
- pip
- Internet connection (to download packages and run Gradio)

### 🐍 Virtual Environment Setup (Recommended)

```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

Install Required Packages

pip install -r requirements.txt


Running the Application

python gradio_emotion_app1.py
Then open the local URL (http://127.0.0.1:7860/) in your browser.

🧠 Model Info
Model Type: ONNX

Trained On: FER2013 dataset

Input Size: 48x48 grayscale face image

Output: Probability scores for 7 emotion classes

💬 Chatbot Behavior
Recognizes:
Greetings (hi, hello, hey)

Moods (sad, happy, anxious, bored, lonely, depressed)

Responses (yes, no, thank you, okay)

Replies With:
Motivational quotes

Empathetic responses

Mental wellness tips like breathing, gratitude, music, and more

🧪 Sample Prompts
“I'm feeling sad”

“Give me a quote”

“I’m anxious today”

“Hi”

“I feel lonely”

📸 Example Use Cases
Check your emotional state with a webcam or photo

Have a calming chat after a stressful day

Receive uplifting quotes and wellness activities

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Emotion model based on FER2013 dataset

Gradio for UI

ONNX Runtime for inference

🔒 Data & Privacy
This app requires internet to function, but no user data or images are sent to any third-party server intentionally. You should still use with caution and avoid uploading sensitive personal data.

💬 Contact
📧 Email: vinuthahallur1709@gmail.com

🌐 GitHub: https://github.com/Vinutha-17

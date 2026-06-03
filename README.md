# hexis
A real-time computer vision project that enables users to control a computer using hand gestures captured through a webcam. The system detects hand landmarks and maps them to different actions such as mouse movement, clicks, scrolling, and custom controls.

---

## 🚀 Features

- ✋ Real-time hand tracking using webcam
- 🧠 Gesture recognition using machine learning landmarks
- 🖱️ Mouse cursor control via hand movement
- 👆 Click detection using finger gestures
- 🔊 Volume / media control mode (optional)
- 🎮 Custom gesture mapping for games or apps
- ♿ Accessibility mode for hands-free interaction
- 🔌 Modular system for adding new “abilities”

---

## 🧠 How It Works

1. Captures video feed from webcam
2. Detects hand using a trained landmark detection model
3. Extracts 21 key hand points (fingertips, joints, palm)
4. Interprets gestures based on finger positions and angles
5. Maps gestures to system actions (mouse, keyboard, etc.)

---

## 🛠️ Tech Stack

- Python 🐍
- OpenCV (computer vision)
- MediaPipe (hand tracking model)
- NumPy (data processing)
- PyAutoGUI (mouse/keyboard control)

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-username/hand-gesture-control.git

# Navigate into project
cd hand-gesture-control

# Install dependencies
pip install -r requirements.txt

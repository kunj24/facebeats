# FaceBeats 🎭🎵

**Real-Time Facial Emotion Detection with Music Playlist Recommendation**

## 📌 Project Overview
FaceBeats is an AI-powered real-time facial emotion detection system that captures emotions from a webcam and recommends personalized YouTube Music playlists based on the detected mood.

## 🚀 Features
- Detects emotions in real time using OpenCV and a CNN model.
- Supports five emotion categories: **Angry, Happy, Neutral, Sad, Surprise**.
- Uses a trained deep learning model for accurate predictions.
- Automatically plays a YouTube Music playlist based on the most frequent detected mood.
- Runs for **12 seconds**, collects predictions, and then redirects to a mood-based playlist.

## 🏗️ Tech Stack
- **Python** (Main Programming Language)
- **TensorFlow/Keras** (Deep Learning Model)
- **OpenCV** (Face Detection & Image Processing)
- **Scikit-learn** (Data Processing & Splitting)
- **NumPy & Pandas** (Data Handling)
- **TextBlob** (Sentiment Analysis for Emotion Text Mapping)
- **WebBrowser** (Redirecting to YouTube Music Playlists)

## 📂 Folder Structure
```
FaceBeats/
│── dataset/                  # Dataset (Emotion-wise folder structure)
│── model/                    # Trained Model
│── real_time_facial_emotion_detection/
│   ├── model.h5              # Pre-trained model
│   ├── history.pkl           # Training history
│── main.py                   # Main emotion detection script
│── playlist.py               # Music recommendation script
│── README.md                 # Project Documentation
```

## 🔧 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kunj24/FaceBeats.git
cd FaceBeats
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Emotion Detection
```bash
python gardioo.py
```

## 🎵 Emotion-Based YouTube Playlists
FaceBeats maps detected emotions to predefined **YouTube Music playlists**:
| Emotion  | YouTube Playlist Link |
|----------|-----------------------|
| Happy    | [Click Here](https://music.youtube.com/playlist?list=PLUj1j3bk-sj47GdIpsKdTNtauKIaqhEqQ) |
| Sad      | [Click Here](https://music.youtube.com/playlist?list=PLSdoVPM5WmpH-85UetGgAcifXdEHkIEeD) |
| Neutral  | [Click Here](https://music.youtube.com/playlist?list=PLfP6i5T0-DkIQ9kVo7HCRMjJq3KyW5sRl) |
| Angry    | [Click Here](https://music.youtube.com/playlist?list=PLrbwxlLsQTFOTXHbdyMnpFhx20W5ZugrC) |
| Relaxed  | [Click Here](https://music.youtube.com/playlist?list=PLRbjoNhQ1PaE6FjCrNiWrBCX5tI7T_Can) |

## 🛠️ Training Your Own Model
To train your own model with a custom dataset:
```bash
python train.py
```
This will load images from the dataset folder, preprocess them, and train a **CNN model**.

## 🎥 Demo
![image](https://github.com/user-attachments/assets/7bee1734-c163-43e3-9884-703559db1b0a)


## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📩 Contact
For any queries, reach out at **your.email@example.com** or open an issue.

---
✨ **FaceBeats: Detect, Feel, Listen!** 🎶


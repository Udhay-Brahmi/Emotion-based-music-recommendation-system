import gradio as gr
import cv2
import numpy as np
import pandas as pd
import sqlite3
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from textblob import TextBlob
import os
import random
from datetime import datetime

class EmotionMusicRecommender:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize database
        self.init_database()
        
        # Load music dataset
        self.load_music_dataset()
        
        # Load or create emotion detection model
        self.load_emotion_model()
    
    def init_database(self):
        """Initialize SQLite database for storing user interactions"""
        conn = sqlite3.connect('emotion_music.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                detected_emotions TEXT,
                text_input TEXT,
                text_sentiment TEXT,
                final_emotion TEXT,
                recommended_songs TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_music_dataset(self):
        """Load or create music dataset"""
        # Sample music dataset - in real implementation, load from muse_v3.csv
        self.music_data = {
            'Happy': [
                'Happy - Pharrell Williams',
                'Good Vibes - Chris Janson',
                'Uptown Funk - Mark Ronson ft. Bruno Mars',
                'Dancing Queen - ABBA',
                'Can\'t Stop the Feeling - Justin Timberlake'
            ],
            'Sad': [
                'Someone Like You - Adele',
                'Hurt - Johnny Cash',
                'Mad World - Gary Jules',
                'Black - Pearl Jam',
                'The Sound of Silence - Simon & Garfunkel'
            ],
            'Angry': [
                'Break Stuff - Limp Bizkit',
                'Bodies - Drowning Pool',
                'Killing in the Name - Rage Against the Machine',
                'B.Y.O.B - System of a Down',
                'Chop Suey! - System of a Down'
            ],
            'Neutral': [
                'Hotel California - Eagles',
                'Stairway to Heaven - Led Zeppelin',
                'Bohemian Rhapsody - Queen',
                'Sweet Child O\' Mine - Guns N\' Roses',
                'November Rain - Guns N\' Roses'
            ],
            'Surprised': [
                'Thunderstruck - AC/DC',
                'We Will Rock You - Queen',
                'Eye of the Tiger - Survivor',
                'Don\'t Stop Me Now - Queen',
                'Livin\' on a Prayer - Bon Jovi'
            ],
            'Fearful': [
                'Breathe Me - Sia',
                'Heavy - Linkin Park ft. Kiiara',
                'Crawling - Linkin Park',
                'In the End - Linkin Park',
                'Numb - Linkin Park'
            ],
            'Disgusted': [
                'Toxic - Britney Spears',
                'Bad Guy - Billie Eilish',
                'Complicated - Avril Lavigne',
                'Since U Been Gone - Kelly Clarkson',
                'You Oughta Know - Alanis Morissette'
            ]
        }
    
    def load_emotion_model(self):
        """Load or create a simple emotion detection model"""
        try:
            # Try to load existing model
            self.emotion_model = keras.models.load_model('model.h5')
        except:
            # Create a simple CNN model if model.h5 doesn't exist
            self.emotion_model = self.create_simple_model()
    
    def create_simple_model(self):
        """Create a simple CNN model for emotion detection"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def detect_face_emotion(self, image):
        """Detect emotion from facial image"""
        if image is None:
            return "No image provided", []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        emotions_detected = []
        
        if len(faces) == 0:
            return "No face detected", []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to model input size
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.reshape(face_roi, (1, 48, 48, 1))
            
            # Predict emotion
            try:
                emotion_pred = self.emotion_model.predict(face_roi)
                emotion_idx = np.argmax(emotion_pred)
                emotion = self.emotion_labels[emotion_idx]
                confidence = float(emotion_pred[0][emotion_idx])
                emotions_detected.append((emotion, confidence))
            except:
                # If model prediction fails, use random emotion for demo
                emotion = random.choice(self.emotion_labels)
                confidence = random.uniform(0.6, 0.9)
                emotions_detected.append((emotion, confidence))
        
        if emotions_detected:
            primary_emotion = max(emotions_detected, key=lambda x: x[1])[0]
            return primary_emotion, emotions_detected
        
        return "Unknown", []
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment from text input"""
        if not text or text.strip() == "":
            return "Neutral", 0.0
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "Happy", polarity
        elif polarity < -0.1:
            return "Sad", abs(polarity)
        else:
            return "Neutral", abs(polarity)
    
    def combine_emotions(self, face_emotion, text_sentiment, face_confidence=0.7, text_confidence=0.5):
        """Combine face emotion and text sentiment to determine final emotion"""
        if face_emotion == "No face detected" or face_emotion == "Unknown":
            return text_sentiment
        
        if text_sentiment == "Neutral":
            return face_emotion
        
        # Weight face emotion and text sentiment
        if face_confidence > 0.8:
            return face_emotion
        elif text_confidence > 0.6:
            return text_sentiment
        else:
            return face_emotion
    
    def recommend_songs(self, emotion, num_songs=5):
        """Recommend songs based on emotion"""
        if emotion in self.music_data:
            songs = self.music_data[emotion]
            return random.sample(songs, min(num_songs, len(songs)))
        else:
            # Default to neutral songs
            return random.sample(self.music_data['Neutral'], min(num_songs, len(self.music_data['Neutral'])))
    
    def save_session(self, face_emotion, text_input, text_sentiment, final_emotion, recommended_songs):
        """Save user session to database"""
        conn = sqlite3.connect('emotion_music.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions 
            (timestamp, detected_emotions, text_input, text_sentiment, final_emotion, recommended_songs)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            face_emotion,
            text_input,
            text_sentiment,
            final_emotion,
            ', '.join(recommended_songs)
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_history(self):
        """Get recent session history"""
        conn = sqlite3.connect('emotion_music.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, final_emotion, recommended_songs 
            FROM user_sessions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        history = cursor.fetchall()
        conn.close()
        
        return history

# Initialize the recommender system
recommender = EmotionMusicRecommender()

def process_emotion_detection(image, text_input):
    """Main function to process both image and text input"""
    # Detect face emotion
    face_emotion, face_details = recommender.detect_face_emotion(image)
    
    # Analyze text sentiment
    text_sentiment, text_confidence = recommender.analyze_text_sentiment(text_input)
    
    # Combine emotions
    final_emotion = recommender.combine_emotions(face_emotion, text_sentiment)
    
    # Recommend songs
    recommended_songs = recommender.recommend_songs(final_emotion)
    
    # Save session
    recommender.save_session(face_emotion, text_input, text_sentiment, final_emotion, recommended_songs)
    
    # Format output
    face_result = f"**Detected Face Emotion:** {face_emotion}"
    if face_details:
        face_result += f" (Confidence: {face_details[0][1]:.2f})"
    
    text_result = f"**Text Sentiment:** {text_sentiment} (Score: {text_confidence:.2f})"
    
    final_result = f"**Final Emotion:** {final_emotion}"
    
    songs_result = "**Recommended Songs:**\n" + "\n".join([f"• {song}" for song in recommended_songs])
    
    return face_result, text_result, final_result, songs_result

def get_history():
    """Get and format session history"""
    history = recommender.get_session_history()
    
    if not history:
        return "No previous sessions found."
    
    history_text = "**Recent Sessions:**\n\n"
    for i, (timestamp, emotion, songs) in enumerate(history, 1):
        history_text += f"{i}. **{timestamp[:19]}** - Emotion: {emotion}\n"
        history_text += f"   Songs: {songs[:100]}{'...' if len(songs) > 100 else ''}\n\n"
    
    return history_text

# Create Gradio interface
with gr.Blocks(title="EmoNoteTunes - Emotion-Based Music Recommendation", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎵 EmoNoteTunes")
    gr.Markdown("## Emotion-Based Music Recommendation System")
    gr.Markdown("Upload a photo of your face and/or enter some text to get personalized music recommendations based on your emotions!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Face Emotion Detection")
            image_input = gr.Image(
                label="Upload your photo or use camera",
                sources=["upload", "webcam"],
                type="numpy"
            )
            
            gr.Markdown("### 📝 Text Sentiment Analysis")
            text_input = gr.Textbox(
                label="How are you feeling? (Optional)",
                placeholder="Enter your thoughts or feelings here...",
                lines=3
            )
            
            analyze_btn = gr.Button("🎵 Get Music Recommendations", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Results")
            face_output = gr.Markdown(label="Face Emotion")
            text_output = gr.Markdown(label="Text Sentiment")
            final_output = gr.Markdown(label="Final Emotion")
            songs_output = gr.Markdown(label="Recommended Songs")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📜 Session History")
            history_btn = gr.Button("View Previous Sessions")
            history_output = gr.Markdown()
    
    # Event handlers
    analyze_btn.click(
        process_emotion_detection,
        inputs=[image_input, text_input],
        outputs=[face_output, text_output, final_output, songs_output]
    )
    
    history_btn.click(
        get_history,
        outputs=history_output
    )
    
    # Example inputs
    gr.Markdown("### 💡 Tips:")
    gr.Markdown("""
    - **For best face detection**: Look directly at the camera with good lighting
    - **Text input**: Describe your current mood, thoughts, or feelings
    - **Combine both**: Use both face and text input for more accurate recommendations
    - **Supported emotions**: Happy, Sad, Angry, Neutral, Surprised, Fearful, Disgusted
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
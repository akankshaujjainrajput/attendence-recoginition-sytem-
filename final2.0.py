import cv2
import numpy as np
import tensorflow as tf
import json
import random
import time
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
model = tf.keras.models.load_model("model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Assign emojis to emotions
emoji_dict = {
    'Happy': '😊', 'Sad': '😢', 'Angry': '😡', 'Surprise': '😲',
    'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
}

# Load font for emojis (Ensure you have it)
try:
    emoji_font = ImageFont.truetype("seguiemj.ttf", 30)  # Windows font
except:
    emoji_font = None  # Fallback if font is missing

# Store falling emojis
falling_emojis = []

import cv2
import numpy as np

def draw_rounded_rectangle(img, top_left, bottom_right, radius, color, border_color, border_thickness):
    """
    Draws a rounded rectangle with an optional border using a mask.
    
    :param img: OpenCV image
    :param top_left: (x1, y1) coordinates of the top-left corner
    :param bottom_right: (x2, y2) coordinates of the bottom-right corner
    :param radius: Radius of the rounded corners
    :param color: (B, G, R) color of the filled rectangle
    :param border_color: (B, G, R) color of the border
    :param border_thickness: Thickness of the border
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1

    # Create a blank mask
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw a filled rounded rectangle on the mask
    rect_color = color if border_thickness < 0 else border_color
    cv2.rectangle(mask, (radius, 0), (w - radius, h), rect_color, -1)  # Top and bottom
    cv2.rectangle(mask, (0, radius), (w, h - radius), rect_color, -1)  # Left and right
    cv2.circle(mask, (radius, radius), radius, rect_color, -1)  # Top-left
    cv2.circle(mask, (w - radius, radius), radius, rect_color, -1)  # Top-right
    cv2.circle(mask, (radius, h - radius), radius, rect_color, -1)  # Bottom-left
    cv2.circle(mask, (w - radius, h - radius), radius, rect_color, -1)  # Bottom-right

    # Paste the rounded rectangle onto the original frame
    img[y1:y2, x1:x2] = cv2.addWeighted(img[y1:y2, x1:x2], 1, mask, 1, 0)

    # If border is needed, draw a slightly smaller rounded rectangle inside
    if border_thickness > 0:
        draw_rounded_rectangle(img, (x1 + border_thickness, y1 + border_thickness),
                               (x2 - border_thickness, y2 - border_thickness),
                               radius, color, None, -1)


# Function to load quotes
def load_quotes():
    with open("quotes.json", "r") as file:
        return json.load(file)

# Function to display quotes based on emotion
def display_quotes(emotion):
    quotes = load_quotes()
    return random.choice(quotes.get(emotion, ["No quotes available."]))

# Function to update and draw falling emojis
def update_emoji_shower(frame, frame_width, frame_height, emoji):
    global falling_emojis

    # Generate new emojis randomly
    if random.random() < 0.2:  # Adjust probability to control emoji spawn rate
        new_emoji = {
            'x': random.randint(0, frame_width - 30),
            'y': 0,  # Start at the top
            'emoji': emoji
        }
        falling_emojis.append(new_emoji)

    # Move emojis downward
    for emoji in falling_emojis:
        emoji['y'] += 5  # Falling speed

    # Remove emojis that have fallen off-screen
    falling_emojis[:] = [e for e in falling_emojis if e['y'] < frame_height]

    # Convert OpenCV frame to PIL image for proper emoji rendering
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Draw emojis on the frame
    for emoji in falling_emojis:
        if emoji_font:
            draw.text((emoji['x'], emoji['y']), emoji['emoji'], font=emoji_font, fill=(255, 255, 255))

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to run real-time emotion detection with emoji shower
def real_time_emotion_detection():
    cap = cv2.VideoCapture(1)
    last_capture_time = time.time()
    emotion = "Detecting..."
    quote = ""
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face_rgb = face_rgb / 255.0
            face_rgb = np.expand_dims(face_rgb, axis=0)

            if time.time() - last_capture_time >= 4:
                last_capture_time = time.time()
                prediction = model.predict(face_rgb)
                emotion = emotion_labels[np.argmax(prediction)]
                quote = display_quotes(emotion)

        # Draw rectangle and labels
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add falling emojis
        frame = update_emoji_shower(frame, frame_width, frame_height, emoji_dict.get(emotion, "🚀"))

        # Draw a black box for the quote
        # Light gray box with dark gray border
        draw_rounded_rectangle(frame, 
                       (20, frame_height - 50), (frame_width-20, frame_height-10), 
                       radius=15, 
                       color=(220,220,220),  # Light gray fill
                       border_color=(105, 105, 105),  # Dark gray border
                       border_thickness=3)

        cv2.putText(frame, quote, (35, frame_height - 25), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow('Emotion Detection with Emoji Shower', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the emotion detection with emoji shower
real_time_emotion_detection()

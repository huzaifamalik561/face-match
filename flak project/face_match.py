from flask import Flask, render_template, request
import cv2
import numpy as np
import face_recognition
from mtcnn.mtcnn import MTCNN
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    try:
        id_card_image = request.files['id_card_image']
        user_image = request.files['user_image']

        # Load and encode the ID card image
        id_card_image = face_recognition.load_image_file(id_card_image)
        id_card_encoding = face_recognition.face_encodings(id_card_image)[0]

        # Load and encode the user input image
        user_input_image = user_image.read()
        img = cv2.imdecode(np.frombuffer(user_input_image, np.uint8), -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        user_input_encoding = face_recognition.face_encodings(img_rgb)[0]

        # Calculate the distance between the eyes for ID card image
        landmarks_id_card = MTCNN().detect_faces(id_card_image)[0]['keypoints']
        left_eye_id_card = landmarks_id_card['left_eye']
        right_eye_id_card = landmarks_id_card['right_eye']
        distance_id_card = np.linalg.norm(np.array(left_eye_id_card) - np.array(right_eye_id_card))

        # Calculate the distance between the eyes for user image
        landmarks_user = MTCNN().detect_faces(img_rgb)[0]['keypoints']
        left_eye_user = landmarks_user['left_eye']
        right_eye_user = landmarks_user['right_eye']
        distance_user = np.linalg.norm(np.array(left_eye_user) - np.array(right_eye_user))

        # Compare face encodings
        similarity_threshold = 0.5
        face_distance = face_recognition.face_distance([id_card_encoding], user_input_encoding)[0]

        # Extract faces from images
        face_locations_id_card = face_recognition.face_locations(id_card_image)
        face_locations_user = face_recognition.face_locations(img_rgb)

        id_card_faces = []
        for (top, right, bottom, left) in face_locations_id_card:
            face = id_card_image[top:bottom, left:right]
            id_card_faces.append(face)

        user_faces = []
        for (top, right, bottom, left) in face_locations_user:
            face = img_rgb[top:bottom, left:right]
            user_faces.append(face)

        # Convert face images to base64 for displaying
        id_card_faces_base64 = []
        for face in id_card_faces:
            _, face_buffer = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(face_buffer).decode('utf-8')
            id_card_faces_base64.append(face_base64)

        user_faces_base64 = []
        for face in user_faces:
            _, face_buffer = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(face_buffer).decode('utf-8')
            user_faces_base64.append(face_base64)

        if face_distance <= similarity_threshold:
            result = "ID card and user image match."
        else:
            result = "ID card and user image do not match."

        return render_template('index.html', result=result, id_card_faces=id_card_faces_base64, user_faces=user_faces_base64, distance_id_card=distance_id_card, distance_user=distance_user)
    except Exception as e:
        return render_template('index.html', result="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
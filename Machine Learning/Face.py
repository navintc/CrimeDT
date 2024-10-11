import os
import cv2
import numpy as np
import math
import time
from flask import Blueprint, jsonify, Response, stream_with_context
import face_recognition

face_recognition_bp = Blueprint('face_recognition', __name__)

def face_confidence(face_distance, face_match_threshold=0.2):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_image_paths = []  # To store image paths
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image_path = f'D:/Research/Machine Learning/faces/{image}'
            face_image = face_recognition.load_image_file(face_image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            self.known_face_image_paths.append(face_image_path)  # Store the path

    def recognize_face_in_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        face_image_paths = []  # List to store matching image paths

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'unknown'
            confidence = 'unknown'
            image_path = ''

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                image_path = self.known_face_image_paths[best_match_index]  # Get matching image path

            face_names.append(name)
            face_image_paths.append(image_path)  # Append the path to the list

        return face_names, face_image_paths


face_recognition_instance = FaceRecognition()


@face_recognition_bp.route('/recognize', methods=['GET'])
def recognize():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return jsonify({"error": "Webcam not found"}), 404

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Recognize faces in the frame
        matching_faces, matching_images = face_recognition_instance.recognize_face_in_frame(frame)

        if matching_faces and 'unknown' not in matching_faces:
            # Send the first match found
            response = {
                "faces": matching_faces,
                "images": matching_images
            }
            video_capture.release()
            return jsonify(response)  # Send only the first match response

        # Display the video feed with detected faces (optional for debugging)
        for (top, right, bottom, left) in face_recognition.face_locations(frame):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        # Press 'q' to quit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "No image found"}), 404  # Default case

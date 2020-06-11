from skimage import io
import dlib
import sys
import os
import cv2
from matplotlib import pyplot as plt
# from IPython import Image, display
import face_recognition as fr
import tensorflow as tf


# Req 4. 이미지(Encoder) 모델 구현하기
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


def face_detection():
    image_path = "./testimage/seul.jpg"

    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)

    for(top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    plt.rcParams["figure.figsize"] = (16, 16)
    plt.imshow(image)
    plt.show()


# 이미지에서 얼굴의 랜드마크 찾기

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "./models/shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
# file_name = sys.argv[1]
file_name = "./testimage/jung.png"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
# face_aligner = openface.AlignDlib(predictor_model)

win = dlib.image_window()

# Load the image
# image = io.imread(file_name)
image = cv2.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(
    len(detected_faces), file_name))

# Show the desktop window with the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i,
                                                                             face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)

    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # Draw the face landmarks on the screen.
    win.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()

# # Loop through each face we found in the image
# for i, face_rect in enumerate(detected_faces):

#         # Detected faces are returned as an object with the coordinates
#         # of the top, left, right and bottom edges
#     print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i,
#                                                                              face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

#     # Get the the face's pose
#     pose_landmarks = face_pose_predictor(image, face_rect)

#     # Use openface to calculate and perform the face alignment
#     alignedFace = face_aligner.align(
#         534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

#     # Save the aligned image to a file
#     cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)

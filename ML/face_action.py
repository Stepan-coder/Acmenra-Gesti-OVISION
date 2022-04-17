import os
import cv2
import time
import math
import numpy as np
from face import *
from face_mesh import *
from pyagender import PyAgender
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
"""pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f"""


class FaceDetection:
    def __init__(self,
                 prototxt_path: str,
                 weights_path: str,
                 mask_model_path: str):
        self.__face_net = cv2.dnn.readNet(prototxt_path, weights_path)
        self.__mask_net = load_model(mask_model_path)
        self.__face_mesh = FaceMeshDetector(max_num_faces=1)
        self.__agender = PyAgender()

    def face_detection(self, frame):
        height, width, channels = frame.shape
        self.__face_net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0)))
        detections = self.__face_net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                left, top, right, bottom = box.astype("int")
                horizontal_delta = int((right - left) * 0.15)
                vertical_delta = int((bottom - top) * 0.20)
                left = left - horizontal_delta if left - horizontal_delta > 0 else 0
                top = top - vertical_delta if top - vertical_delta > 0 else 0
                right = right + horizontal_delta if right + horizontal_delta < width - 1 else width - 1
                bottom = bottom + vertical_delta if bottom + vertical_delta < height - 1 else height - 1
                faces.append(Face(id=i, frame=frame[top:bottom, left:right],
                                  left=left, top=top, right=right, bottom=bottom))
        return faces

    def face_mask_detection(self, face: Face) -> Face:
        frame = cv2.cvtColor(face.get_frame(), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = img_to_array(frame)
        frame = preprocess_input(frame)
        frame = np.array([frame], dtype="float32")
        mask_prediction = self.__mask_net.predict(frame, batch_size=32)
        face.set_is_mask(is_mask=mask_prediction[0][0] > mask_prediction[0][1])
        return face

    def face_mesh_detection(self, face: Face) -> Face:
        face_mesh = self.__face_mesh.find_face_mesh(img=face.get_frame())
        if len(face_mesh) == 1:
            face.set_mesh_points(mesh_points=face_mesh['face_0']['face_points'])
            face.set_mesh_frame(frame=self.__face_mesh.show_faces(img=face.get_frame(), faces=face_mesh))
        return face

    def get_face_distance(self, face: Face) -> Face:
        face_points = face.get_mesh_points()
        point_145 = face_points[145]
        point_374 = face_points[374]
        length = math.hypot(point_374['x'] - point_145['x'], point_374['y'] - point_145['y'])
        face.set_distance(distance=(6.4 * 840) / length)
        return face

    def get_face_agender(self, face: Face) -> Face:
        face_info = self.__agender.detect_genders_ages(face.get_frame())
        try:
            face.set_age(age=round(face_info[0]['age']))
            face.set_gender(gender="women" if face_info[0]['gender'] > 0.5 else "men")
        except:
            pass
        return face







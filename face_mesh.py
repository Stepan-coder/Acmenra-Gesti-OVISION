import cv2
import mediapipe as mp
from typing import Any, List, Dict


class FaceMeshDetector:
    def __init__(self,
                 max_num_faces: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        """
        :param max_num_faces: Max count of detecting faces
        :param min_detection_confidence: coefficient of face detection. Recommended value 0.5
        :param min_tracking_confidence: coefficient of face detection. Recommended value 0.5
        """
        self.__mediapipe_face_mesh = mp.solutions.face_mesh
        self.__mediapipe_draw = mp.solutions.drawing_utils
        self.__drawing_spec = self.__mediapipe_draw.DrawingSpec(thickness=1, circle_radius=1)
        self.__face_mesh_musk = mp.solutions.face_mesh
        self.__face_mesh = self.__mediapipe_face_mesh.FaceMesh(max_num_faces=max_num_faces,
                                                               min_detection_confidence=min_detection_confidence,
                                                               min_tracking_confidence=min_tracking_confidence)

    def findFaceMesh(self, img) -> Dict[str, Dict[int, Dict[str, Any]] or Any]:
        """
        This method finds the faces in the image and returns their parameters
        :param img:
        :return: face landmarks and points coordinates
        example:
        'face_0':
            'face_points':
                0:
                 'x': 10,
                 'y': 104,
                 'normal_x': 0.6319661140441895,
                 'normal_y': 0.8375985026359558,
                 'normal_z': -0.015897102653980255
            'face_landmark':
                Used for automatic face grid construction
        """
        results = self.__face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        faces = {}
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                face = {}
                for id, lm in enumerate(face_landmark.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face[id] = {"x": x, "y": y, "normal_x": lm.x, "normal_y": lm.y, "normal_z": lm.z}
                faces[f"face_{len(faces)}"] = {"face_points": face, "face_landmark": face_landmark}
        return faces

    def show_faces(self, img, faces: Dict[str, Dict[int, Dict[str, Any]] or Any]):
        """
        This method shows faces mesh from face mesh detector
        :param img: input image
        :param faces: information about faces on the frame
        :return: Image, with an overlay of a grid of faces
        """
        for face in faces:
            self.__mediapipe_draw.draw_landmarks(img,
                                                 faces[face]['face_landmark'],
                                                 self.__face_mesh_musk.FACE_CONNECTIONS,
                                                 self.__drawing_spec,
                                                 self.__drawing_spec)
        return img




cap = cv2.VideoCapture(0)
pTime = 0
detector = FaceMeshDetector()
while True:
    success, img = cap.read()
    faces = detector.findFaceMesh(img)
    img = detector.show_faces(img, faces)
    cv2.imshow("Image", img)
    cv2.waitKey(1)




from face_action import *

prototxt_path = r"models\deploy.prototxt"
weights_path = r"models\res10_300x300_ssd_iter_140000.caffemodel"
mask_model_path = r"models\mask_detector.model"
face_recognition = FaceDetection(prototxt_path=prototxt_path,
                                 weights_path=weights_path,
                                 mask_model_path=mask_model_path)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while True:
    ret, frame = camera.read()
    height, width, channels = frame.shape
    people = face_recognition.face_detection(frame=frame)
    frame = np.zeros((height, width, 3), np.uint8)
    for i in range(len(people)):
        people[i] = face_recognition.face_mesh_detection(face=people[i])
        people[i] = face_recognition.get_face_distance(face=people[i])
        people[i] = face_recognition.face_mask_detection(face=people[i])
        people[i] = face_recognition.get_face_agender(face=people[i])
        left, top, right, bottom = people[i].get_coordinates()
        frame[top:bottom, left:right] = people[i].get_frame()
        frame = cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imshow("Face", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
camera.release()
from flask import Flask, render_template, Response
import cv2
from ML.face import Face
from ML.face_action import FaceDetection

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def update_image(img, show_mesh:bool, med_mask:bool, distance:bool, gender:bool, age:bool):
    
    face_detect
    face = Face()
    
    med_mask = "Не отслеживается"
    distance = "Не отслеживается"
    gender = "Не отслеживается"
    age = "Не отслеживается"

    if show_mesh:
        pass

    elif med_mask:
        if face.get_is_mask():
            med_mask = "Надета"
        else:
            med_mask = "Отсутствует"
        
    elif distance:
        distance = face.get_distance()

    elif gender:
        gender = face.get_gender()

    elif age:
        age = face.get_age()

    json = {
        "mask": med_mask,
        "distance": distance,
        "gender": gender,
        "age": age
    }
    return (img, json)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True)
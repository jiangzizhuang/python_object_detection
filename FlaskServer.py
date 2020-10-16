from flask import Flask
from flask import request
from YoloObjectDetection import YOD
import cv2
import json

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def get_image():
    image = request.files.get('image')
    file_name = image.filename
    path = './images/' + file_name
    image.save(path)

    yod = YOD('resource/yolov3-320.cfg', 'resource/yolov3-320.weights')
    res = yod.detect_image(cv2.imread(path))

    return json.dumps(res)

if __name__ == "__main__":
    app.run(debug=True)

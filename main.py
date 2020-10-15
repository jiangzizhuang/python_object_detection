from YoloObjectDetection import YOD

if __name__ == '__main__':
    yod = YOD('resource/yolov3-320.cfg', 'resource/yolov3-320.weights')
    yod.detect_video('hello')
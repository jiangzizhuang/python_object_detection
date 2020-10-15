1) git clone https://github.com/jiangzizhuang/python_object_detection.git
2) cd python_object_detection
3) cd resources
4) wget https://pjreddie.com/media/files/yolov3.weights

Useage:

yod = YOD(yolo_model_cfg, yolo_model_weights)
yod.detect_video(window_name)
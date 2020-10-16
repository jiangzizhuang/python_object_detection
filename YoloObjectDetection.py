import cv2
import numpy as np

class YOD:
    def __init__(self, model_config, model_weights):
        self.whT = 320
        self.thresold = 0.5
        self.nms_thresold = 0.3
        self.classifications = self.get_classification_names()
        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layerNames = self.net.getLayerNames()
        self.outputLayerNames = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def get_classification_names(self):
        with open('resource/coco.names', 'r') as f:
            classifications = f.read().rstrip('\n').split('\n')
            return classifications

    def get_network_outputs(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.whT, self.whT), (0, 0, 0), swapRB=1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.outputLayerNames)
        return outputs

    def detect_object(self, outputs, image):
        i_h, i_w, _ = image.shape
        boxes = []
        classIds = []
        confidences = []
        objs = []
        for output in outputs:
            for det in output:
                w, h = int(i_w * det[2]), int(i_h * det[3])
                x, y = int(det[0] * i_w - w / 2), int(det[1] * i_h - h / 2)
                scores = det[5:]
                classId = np.argmax(scores)
                score = scores[classId]
                if score > self.thresold:
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidences.append(float(score))
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.thresold, nms_threshold=self.nms_thresold)
        for i in indices:
            i = i[0]
            obj = dict()
            obj['box'] = boxes[i]
            obj['classification'] = self.classifications[classIds[i]]
            obj['confidence'] = int(confidences[i] * 100)
            objs.append(obj)

        return objs

    def process_image(self, image, objs):
        for obj in objs:
            x, y, w, h = obj['box'][0], obj['box'][1], obj['box'][2], obj['box'][3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (244, 244, 0), 2)
            cv2.putText(image, '{0} {1}%'.format(obj['classification'], obj['confidence']), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (230, 100, 10), 2)

    def detect_video(self, window_name = 'Camera'):
        cam = cv2.VideoCapture(0)
        while True:
            _, image = cam.read()
            outputs = self.get_network_outputs(image)
            objs = self.detect_object(outputs, image)
            self.process_image(image, objs)
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

    def detect_image(self, image):
        outputs = self.get_network_outputs(image)
        objs = self.detect_object(outputs, image)
        return objs
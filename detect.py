import cv2
import numpy as np

def load_yolo():
    # Load YOLOv3-tiny model
    net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")
    layer_names = net.getLayerNames()
    # getUnconnectedOutLayers() mengembalikan index layer yang jadi output
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_people(img, net, output_layers, conf_threshold=0.2, nms_threshold=0.35):
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=(416, 416),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)

            # COCO: class_id 0 = 'person'
            if class_id == 0:
                confidence = float(scores[class_id])
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # ====== FILTER HEURISTIK UNTUK NGURANGIN TAS ======
                    # 1. Minimal area (supaya box kecil banget nggak kepikih)
                    img_area = width * height
                    box_area = w * h
                    if box_area < 0.01 * img_area:   # < 1% area gambar → skip
                        continue

                    # 2. Aspect ratio: orang biasanya lebih tinggi dari lebar
                    aspect_ratio = h / float(w + 1e-6)
                    if aspect_ratio < 1.1:  # kalau terlalu "melebar", kemungkinan tas / objek lain
                        continue
                    # =================================================

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

    # NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    people_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            people_boxes.append(boxes[i])

    return people_boxes
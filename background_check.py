import cv2
import numpy as np

def background_check(path):
    net = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
    classes = []
    with open("yolov3/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(path)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640),  swapRB=True, crop=False)
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    objects = list()
    conf=list()
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            #x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objects.append(label)
            conf.append(confidence)
            #color = colors[class_ids[i]]
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)
    
    person_count=0
    for object in objects:
        if object=='person':
            person_count=person_count+1
    
    if person_count>1:
        text2='person detected in background'
    else:
        text2='suitable'

    return text2
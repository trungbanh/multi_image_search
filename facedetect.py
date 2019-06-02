import cv2 

def face_detect (path):
    modelFile = "./modelDNN/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "./modelDNN/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    image = cv2.imread('./images/'+path)

    frameWidth = len(image[0])
    frameHeight = len(image)
    conf_threshold =0.4

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),3)

    return image
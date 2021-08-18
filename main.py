import numpy as np #array
import imutils #resize
import cv2 #images&DNN
import time #delay

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2 #Make sure accuracy/confident

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed...")
vs = cv2.VideoCapture(2)
time.sleep(2.0)

while True:
	_,frame = vs.read() #read the frame from camera
	frame = imutils.resize(frame, width=500) #resize for display in ouput
	(h, w) = frame.shape[:2] #obtaining h & w
	imResize = cv2.resize(frame, (300, 300)) #pre process pre train
	blob = cv2.dnn.blobFromImage(imResize,
		0.007843, (300, 300), 127.5) #blob

	net.setInput(blob)
	detections = net.forward()
	print(detections)
	detShape = detections.shape[2]
	for i in np.arange(0,detShape):
		confidence = detections[0, 0, i, 2]
		if confidence > confThresh:     
			idx = int(detections[0, 0, i, 1])
			print(idx)
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			if startY - 15 > 15:
				y = startY - 15
			else:
				y =startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
vs.release()
cv2.destroyAllWindows()


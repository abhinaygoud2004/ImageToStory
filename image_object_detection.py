import cv2
from imread_from_url import imread_from_url

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
# img_url = "/Users/maturiabhinaygoud/Downloads/test2.jpeg"
# img = imread_from_url(img_url)

# Read local image using OpenCV
img_path = "/Users/maturiabhinaygoud/Downloads/test3.jpeg"
img = cv2.imread(img_path)



# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

print()

for box, score, class_id in zip(boxes, scores, class_ids):
    print(f"Class: {class_id}, Confidence: {score:.2f}")
    print(f"Box Coordinates: {box}")
    print("="*30)



# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
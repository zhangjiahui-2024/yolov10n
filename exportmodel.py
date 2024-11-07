from ultralytics import YOLOv10

model = YOLOv10("yolov10n.pt")
model = YOLOv10('runs/detect/train9/weights/best.pt')

model.export(format='onnx')
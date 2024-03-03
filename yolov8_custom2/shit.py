from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO('yolov8s.pt').cuda()
    yolo.train(data= "config.yaml", epochs=7, batch= 32, workers= 2, optimizer= 'Adam', imgsz= 640, lr0= 0.001, task= 'detect')
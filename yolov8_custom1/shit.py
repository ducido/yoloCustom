from ultralytics import YOLO

if __name__ == '__main__':

    yolo = YOLO('yolov8n.pt').cuda()
    yolo.train(data= "config.yaml", epochs=2, batch= 9, workers= 1, optimizer= 'Adam', imgsz= 320, lr0= 0.0001, task= 'detect', amp= False)
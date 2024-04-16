from ultralytics import YOLO


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLO('runs/detect/blurry2/weights/best.pt')

    # Define the path to the images
    source = 'data/val/images'

    model.val(save=True, imgsz=640, data='fish.yaml', batch=16, conf=0.25, iou=0.6, device=0)


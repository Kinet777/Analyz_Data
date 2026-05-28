import cv2
import easyocr
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

class VisionAnalyzer:
    def __init__(self):
        print("Инициализация VisionAnalyzer...")
        
        # Настройка OCR (ПЗ 3)
        self.reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
        
        # Настройка YOLO (ПЗ 5)
        self.yolo_model = YOLO("yolov8n.pt") # Загрузит базовую легковесную модель
        
        # Настройка ResNet (ПЗ 6)
        self.resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_model.eval()
        self.resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Загрузка классов ImageNet для ResNet
        self.imagenet_classes = models.ResNet50_Weights.DEFAULT.meta.get("categories", [])

    def analyze_frame(self, frame_path):
        """Анализирует один кадр тремя моделями."""
        results = {}
        
        # 1. Распознавание текста (OCR)
        text_results = self.reader.readtext(frame_path)
        texts = [text[1] for text in text_results if text[2] > 0.5] # Фильтруем по уверенности > 50%
        results["text"] = texts
        
        # 2. YOLO Детектирование объектов
        yolo_results = self.yolo_model(frame_path, verbose=False)[0]
        objects = []
        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                objects.append(class_name)
        results["yolo_objects"] = objects
        
        # 3. ResNet Классификация сцены
        img = Image.open(frame_path).convert('RGB')
        img_t = self.resnet_transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            out = self.resnet_model(batch_t)
        
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        
        # Берем топ-3 класса
        top_classes = []
        # Если не удалось загрузить классы из meta (бывает в старых версиях torchvision), 
        # просто вернем ID классов. Но в новых все работает.
        try:
            from torchvision.models import ResNet50_Weights
            categories = ResNet50_Weights.DEFAULT.meta["categories"]
            for idx in indices[0][:3]:
                top_classes.append(categories[idx])
        except:
            for idx in indices[0][:3]:
                top_classes.append(f"class_id_{idx.item()}")
                
        results["resnet_scene"] = top_classes
        
        return results

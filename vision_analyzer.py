import os

import easyocr
import torch
from PIL import Image
from torchvision import models
from ultralytics import YOLO


class VisionAnalyzer:
    def __init__(self):
        print("Инициализация VisionAnalyzer...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ocr_confidence_threshold = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.5"))
        self.object_confidence_threshold = float(os.getenv("OBJECT_CONFIDENCE_THRESHOLD", "0.45"))
        self.classifier_top_k = int(os.getenv("IMAGE_CLASSIFIER_TOP_K", "5"))

        # ПЗ 3: OCR для русского и английского текста на кадрах.
        self.reader = easyocr.Reader(["ru", "en"], gpu=torch.cuda.is_available())

        # ПЗ 5: YOLOv8s точнее nano-версии, но все еще подъемен для CPU/слабой ВМ.
        self.object_detector_name = os.getenv("OBJECT_DETECTOR_MODEL", "yolov8s.pt")
        self.object_detector = YOLO(self.object_detector_name)

        # ПЗ 6: современный ImageNet-классификатор кадра.
        self.image_classifier_name = os.getenv("IMAGE_CLASSIFIER_MODEL", "efficientnet_v2_s")
        self.image_classifier, weights = self._load_image_classifier(self.image_classifier_name)
        self.image_classifier.to(self.device)
        self.image_classifier.eval()
        self.image_classifier_transform = weights.transforms()
        self.image_classifier_categories = weights.meta.get("categories", [])

        print(
            "VisionAnalyzer готов: "
            f"OCR=EasyOCR, detector={self.object_detector_name}, "
            f"classifier={self.image_classifier_name}, device={self.device}"
        )

    def _load_image_classifier(self, model_name):
        normalized_name = model_name.lower().replace("-", "_")

        if normalized_name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            return models.efficientnet_v2_s(weights=weights), weights

        if normalized_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            return models.efficientnet_b0(weights=weights), weights

        if normalized_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            return models.mobilenet_v3_large(weights=weights), weights

        raise ValueError(
            "Неизвестный IMAGE_CLASSIFIER_MODEL. "
            "Допустимо: efficientnet_v2_s, efficientnet_b0, mobilenet_v3_large"
        )

    def _analyze_text(self, frame_path):
        text_results = self.reader.readtext(frame_path)
        detections = []
        for _, text, confidence in text_results:
            confidence = float(confidence)
            if confidence >= self.ocr_confidence_threshold:
                detections.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                })
        return detections

    def _detect_objects(self, frame_path):
        yolo_results = self.object_detector(frame_path, verbose=False)[0]
        detections = []

        for box in yolo_results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.object_confidence_threshold:
                continue

            class_id = int(box.cls[0])
            label = self.object_detector.names[class_id]
            xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            detections.append({
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": xyxy,
            })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _classify_image(self, frame_path):
        image = Image.open(frame_path).convert("RGB")
        image_tensor = self.image_classifier_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.image_classifier(image_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            top_probabilities, top_indices = torch.topk(probabilities, self.classifier_top_k)

        predictions = []
        for probability, index in zip(top_probabilities, top_indices):
            class_index = int(index.item())
            label = (
                self.image_classifier_categories[class_index]
                if class_index < len(self.image_classifier_categories)
                else f"class_id_{class_index}"
            )
            predictions.append({
                "label": label,
                "confidence": round(float(probability.item()), 4),
            })

        return predictions

    def analyze_frame(self, frame_path):
        """Анализирует один кадр OCR, объектным детектором и классификатором изображения."""
        ocr_detections = self._analyze_text(frame_path)
        object_detections = self._detect_objects(frame_path)
        classifier_predictions = self._classify_image(frame_path)

        return {
            "text": [item["text"] for item in ocr_detections],
            "ocr_detections": ocr_detections,
            "yolo_objects": [item["label"] for item in object_detections],
            "object_detections": object_detections,
            "image_classifier_labels": [item["label"] for item in classifier_predictions],
            "image_classifier_predictions": classifier_predictions,
            "vision_models": {
                "ocr": "EasyOCR",
                "object_detector": self.object_detector_name,
                "image_classifier": self.image_classifier_name,
            },
        }

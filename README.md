%python
%pip install opencv-python ultralytics
import os
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
import xml.etree.ElementTree as ET
import shutil

class ChequeProcessor:
    """Handles multi-stage dataset preparation"""
    def __init__(self, xml_path, images_dir, output_dir):
        self.xml_path = xml_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.main_classes = ['dollar_amount_section', 'legal_amount_section', 'date_section']
        os.makedirs(os.path.expanduser(output_dir), exist_ok=True)

    def xml_to_yolo_stage1(self):
        """Stage 1: Main object detection dataset"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Create main class mapping
        main_class_map = {cls: idx for idx, cls in enumerate(self.main_classes)}
        
        # Create directories for images and labels
        train_img_dir = os.path.join(self.output_dir, 'stage1', 'images', 'train')
        val_img_dir = os.path.join(self.output_dir, 'stage1', 'images', 'val')
        train_label_dir = os.path.join(self.output_dir, 'stage1', 'labels', 'train')
        val_label_dir = os.path.join(self.output_dir, 'stage1', 'labels', 'val')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # Process annotations
        for image in root.findall('image'):
            img_file = image.get('file')
            if img_file is None:
                continue
            img_path = os.path.join(self.images_dir, img_file)
            if not os.path.exists(img_path):
                continue
            img_width = int(image.get('width'))
            img_height = int(image.get('height'))
            
            # Split data into train and val
            if np.random.rand() < 0.8:
                img_dest = os.path.join(train_img_dir, img_file)
                label_dest = os.path.join(train_label_dir, f"{os.path.splitext(img_file)[0]}.txt")
            else:
                img_dest = os.path.join(val_img_dir, img_file)
                label_dest = os.path.join(val_label_dir, f"{os.path.splitext(img_file)[0]}.txt")
            
            shutil.copy(img_path, img_dest)
            
            with open(label_dest, 'w') as lf:
                for box in image.findall('box'):
                    label = box.get('label')
                    if label in self.main_classes:
                        # Convert bbox to YOLO format
                        x_min = float(box.get('xtl'))
                        y_min = float(box.get('ytl'))
                        x_max = float(box.get('xbr'))
                        y_max = float(box.get('ybr'))
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        class_id = main_class_map[label]
                        lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    def create_yolo_configs(self):
        """Generate YOLO configs for all stages"""
        # Stage 1 config
        stage1_config = {
            'path': os.path.join(self.output_dir, 'stage1'),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.main_classes),
            'names': self.main_classes
        }
        
        # Save config
        with open(os.path.join(self.output_dir, 'stage1_config.yaml'), 'w') as f:
            yaml.dump(stage1_config, f)

class ChequeTrainingPipeline:
    """Orchestrates the multi-stage training"""
    def __init__(self, processor):
        self.processor = processor
        self.stage1_model = None

    def train_stage1(self, epochs=50, batch_size=8):
        """Train main object detection model"""
        config_path = os.path.join(self.processor.output_dir, 'stage1_config.yaml')
        model = YOLO('yolov8s.pt')
        results = model.train(
            data=config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device='cpu'
        )
        self.stage1_model = model
        return results

class ChequeInspector:
    """Handles multi-stage inference with OCR integration"""
    def __init__(self, stage1_model_path, stage2_model_dir):
        self.stage1_model = YOLO(stage1_model_path)
        self.stage2_models = {
            'dollar_amount': YOLO(os.path.join(stage2_model_dir, 'dollar_amount', 'weights', 'best.pt')),
            'legal_amount': YOLO(os.path.join(stage2_model_dir, 'legal_amount', 'weights', 'best.pt')),
            'date': YOLO(os.path.join(stage2_model_dir, 'date', 'weights', 'best.pt'))
        }
        self.date_parser = DateParser()

    def predict(self, image_path):
        """Full pipeline prediction"""
        # Stage 1: Detect main objects
        main_objects = self.stage1_model(image_path)[0]
        
        results = []
        for box in main_objects.boxes:
            cls_id = int(box.cls)
            main_class = self.stage1_model.names[cls_id]
            bbox = box.xyxy[0].tolist()
            confidence = float(box.conf)
            
            # Crop object region
            img = Image.open(image_path)
            cropped = img.crop(bbox)
            
            # Stage 2: Get attributes
            if main_class in ['dollar_amount', 'legal_amount']:
                attrs = self._predict_attributes(cropped, main_class)
                result = {
                    'type': main_class,
                    'attributes': attrs,
                    'confidence': confidence,
                    'bbox': bbox
                }
            elif main_class == 'date':
                date_format = self._predict_date_format(cropped)
                date_value = self.date_parser.parse_date(cropped, date_format)
                result = {
                    'type': 'date',
                    'format': date_format,
                    'value': date_value,
                    'confidence': confidence,
                    'bbox': bbox
                }
            
            results.append(result)
        
        return results

    def _predict_attributes(self, cropped_image, main_class):
        """Predict attributes for dollar/legal amounts"""
        model = self.stage2_models[main_class]
        results = model(cropped_image)
        attrs = []
        for box in results[0].boxes:
            for cls in box.cls.unique():
                attr_name = model.names[int(cls)]
                attrs.append(attr_name)
        return list(set(attrs))

    def _predict_date_format(self, cropped_image):
        """Predict date format from cropped region"""
        results = self.stage2_models['date'](cropped_image)
        return self.stage2_models['date'].names[int(results[0].boxes.cls[0])]

class DateParser:
    """Handles date value extraction using OCR and format"""
    def parse_date(self, image, format):
        """Extract and parse date value"""
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess for OCR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # OCR with Tesseract
        text = pytesseract.image_to_string(thresh, config='--psm 6 digits')
        clean_text = ''.join(c for c in text if c.isdigit() or c in ['/', '-'])
        
        # Parse based on format
        format_pattern = format.split('_')[-1]  # Extract pattern from class name
        return self._apply_format(clean_text, format_pattern)

    def _apply_format(self, text, pattern):
        """Apply detected format to OCR text"""
        try:
            # Implement actual date parsing logic based on format
            # This is simplified example
            if pattern == 'mm/dd/yyyy':
                return f"{text[0:2]}/{text[2:4]}/{text[4:8]}"
            elif pattern == 'dd/mm/yyyy':
                return f"{text[0:2]}/{text[2:4]}/{text[4:8]}"
            elif pattern == 'yyyy-mm-dd':
                return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
            return text
        except:
            return "Unable to parse date"

# Usage Example
if __name__ == "__main__":
    # Data preparation
    images_dir = "/Workspace/Users/mrinalini.vettri@fisglobal.com/yolo_check_training/checks"  # Path to the folder containing the check images
    processor = ChequeProcessor('annotations.xml', images_dir, 'checks')
    processor.xml_to_yolo_stage1()
    processor.create_yolo_configs()

    # Training
    pipeline = ChequeTrainingPipeline(processor)
    stage1_results = pipeline.train_stage1(epochs=50)

    # Inference
    inspector = ChequeInspector(
        stage1_model_path='checks/stage1/weights/best.pt',
        stage2_model_dir='checks/stage2'
    )
    predictions = inspector.predict('3307435490.tif')
    
    for pred in predictions:
        if pred['type'] == 'date':
            print(f"Date: {pred['value']} ({pred['format']})")
        else:
            print(f"{pred['type']}: {pred['attributes']}")

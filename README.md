%python
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import tempfile
from typing import List, Tuple
import json
import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SCENE_CHANGE_THRESHOLD = 30.0  # Frame difference threshold for scene detection

# Initialize CLIP model
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)

class VideoAnalyzer:
    def __init__(self):
        self.kb = self.initialize_knowledge_base()
        
    def initialize_knowledge_base(self):
        # Initialize with empty KB
        kb = faiss.IndexFlatL2(model.config.projection_dim)
        return kb

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scene changes using frame differences"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        scenes = []
        prev_frame = None
        scene_start = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                non_zero_count = np.count_nonzero(diff)
                if non_zero_count > SCENE_CHANGE_THRESHOLD:
                    scenes.append((scene_start, current_time))
                    scene_start = current_time
                    
            prev_frame = gray
            
        scenes.append((scene_start, current_time))
        cap.release()
        return scenes

    def extract_key_frames(self, video_path: str, scenes: List[Tuple[float, float]]) -> List[Image.Image]:
        """Extract middle frame from each scene"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        
        for start, end in scenes:
            middle = (start + end) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, middle * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                
        cap.release()
        return frames

    def extract_features(self, frames: List[Image.Image]) -> np.ndarray:
        """Extract features from a sequence of frames using CLIP"""
        inputs = processor(
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        # Aggregate features from all frames
        features = outputs.cpu().numpy()
        return features

    def add_to_knowledge_base(self, features: np.ndarray):
        """Add features to the FAISS knowledge base"""
        self.kb.add(features)

    def search_similar(self, query_features: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar features in the FAISS knowledge base"""
        distances, indices = self.kb.search(query_features, k)
        return list(zip(indices[0], distances[0]))

    def process_video(self, video_path: str) -> List[dict]:
        """Full video processing pipeline"""
        try:
            print("Starting video analysis...")
            print("Detecting scenes...")
            
            # Scene detection
            scenes = self.detect_scenes(video_path)
            print(f"Found {len(scenes)} scenes")
            
            # Key frame extraction
            key_frames = self.extract_key_frames(video_path, scenes)
            print(f"Extracted {len(key_frames)} key frames")
            
            # Feature extraction
            features = self.extract_features(key_frames)
            print(f"Extracted features with shape: {features.shape}")
            
            # Add features to knowledge base
            self.add_to_knowledge_base(features)
            print("Added features to knowledge base")
            
            # Explanation generation
            explanations = self.generate_explanations(key_frames)
            print(f"Generated explanations: {explanations}")
            
            # Format results
            results = [{
                "start": scene[0],
                "end": scene[1],
                "duration": scene[1] - scene[0],
                "explanation": expl,
                "features": feat.tolist()
            } for scene, expl, feat in zip(scenes, explanations, features)]
            
            # Save results to JSON file
            json_path = video_path.replace(".mp4", ".json")
            with open(json_path, "w") as json_file:
                json.dump(results, json_file, indent=4)
            
            # Print results to output terminal
            print("Final Video Summary:\n")
            for i, result in enumerate(results):
                print(f"Scene {i+1} ({result['duration']:.1f}s)")
                print(f"From {result['start']:.1f}s to {result['end']:.1f}s")
                print(f"Summary: {result['explanation']}\n")
            
            return results
        except Exception as e:
            print(f"Error processing video: {e}")
            return []

if __name__ == "__main__":
    # Path to your video
    video_path = "/Workspace/Users/mrinalini.vettri@fisglobal.com/video_analysis/sonar_embedding/dataset/video2.mp4"
    
    analyzer = VideoAnalyzer()
    
    # Process the video
    results = analyzer.process_video(video_path)
    
    # Print or save the results
    print(f"Results for {video_path}:")
    print(json.dumps(results, indent=4))
    
    # Example of searching for similar features
    if results:
        query_features = np.array(results[0]['features']).reshape(1, -1)
        similar_items = analyzer.search_similar(query_features)
        print(f"Similar items: {similar_items}")

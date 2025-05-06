%python
%pip install faiss-cpu
%pip install opencv-python
import faiss
import numpy as np
import cv2
import yaml
import json
import os
from PIL import Image
from datetime import datetime
from huggingface_hub import InferenceClient
from typing import List, Dict, Tuple

# ====================
# Configuration Loader
# ====================

class WorkflowCreatorConfig:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.video_path = config["video_source"]
        self.output_path = config["output_json"]
        self.frame_interval = config.get("frame_interval", 1)
        self.min_confidence = config.get("min_confidence", 0.8)
        self.models = config["models"]
        self.faiss_config = config.get("faiss", {})
        self.screenshot_dir = config.get("screenshot_dir", "screenshots")

        os.makedirs(self.screenshot_dir, exist_ok=True)

# ====================
# Image Embedding & FAISS
# ====================

class ImageVectorStore:
    def __init__(self, config: WorkflowCreatorConfig):
        self.index = None
        self.embeddings = {}
        self.model = InferenceClient(config.models["embedding"])
        self.dimension = 768  # DINO embedding dimension
        self.similarity_threshold = config.faiss_config.get("similarity_threshold", 0.8)
        
    def initialize_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def add_embedding(self, image: Image.Image, scene_id: str):
        embedding = self._get_embedding(image)
        if self.index is None:
            self.initialize_index()
            
        self.embeddings[scene_id] = embedding
        self.index.add(np.array([embedding]).astype('float32'))
        
    def find_similar(self, image: Image.Image) -> Tuple[str, float]:
        query_embed = self._get_embedding(image)
        distances, indices = self.index.search(
            np.array([query_embed]).astype('float32'), 1)
        
        if indices[0][0] == -1:
            return None, 0.0
            
        scene_id = list(self.embeddings.keys())[indices[0][0]]
        similarity = 1 - (distances[0][0] / self.dimension)
        return (scene_id, similarity) if similarity > self.similarity_threshold else (None, 0.0)
        
    def _get_embedding(self, image: Image.Image) -> List[float]:
        response = self.model.feature_extraction(image)
        return response[0]

# ====================
# UI Analysis Prompts
# ====================

UI_ANALYSIS_PROMPT = """Analyze this application UI screen and provide a detailed structural breakdown in JSON format. 
Include the following sections:
1. Screen Metadata:
   - Screen purpose
   - Screen type (login, dashboard, etc.)
   - Layout type (grid, freeform, etc.)

2. Structural Components:
   - Main sections (menu, toolbar, sidebar, main content, status bar)
   - For each section:
     * Position [x1,y1,x2,y2]
     * Contained components
     * Visual style characteristics
     * Interaction state

3. UI Components:
   - For each component (buttons, inputs, tables, etc.):
     * Component type
     * Position [x1,y1,x2,y2]
     * Text content (if any)
     * Visual properties (color, size, font)
     * Interaction state (enabled/disabled)
     * Associated actions
     * Hierarchy path (e.g., "menu/file/open")

4. Text Elements:
   - All visible text elements not part of components
   - Position and styling for each

5. Visual Hierarchy:
   - Z-index information
   - Modal/dialog boxes
   - Hidden elements

Format the response as valid JSON with the structure:
```json
{
  "workflow": {
    "metadata": {
      "created_at": "2024-03-15T15:30:00.123456",
      "video_source": "video.mp4",
      "config": {...}
    },
    "scenes": [
      {
        "id": "scene_1",
        "screen_type": "dashboard",
        "duration": 12.5,
        "screenshot": "screenshots/scene_1.png",
        "structure": [
          {
            "section": "main_menu",
            "position": [0, 0, 1024, 50],
            "components": ["file_menu", "edit_menu"]
          }
        ],
        "components": [
          {
            "type": "button",
            "position": [100, 200, 150, 225],
            "text": "Save",
            "properties": {
              "color": "#4287f5",
              "font": "Arial 12pt"
            },
            "actions": ["click"]
          }
        ],
        "text_elements": [...],
        "actions": [...]
      }
    ],
    "embeddings": "faiss_index.bin",
    "completion": true
  },
  "statistics": {
    "total_scenes": 5,
    "unique_screens": 3
  }
}"

# ====================
# Frame Analysis
# ====================

class FrameAnalyzer:
    def __init__(self, config: WorkflowCreatorConfig):
        self.client = InferenceClient(config.models["screen_analysis"])
        self.min_confidence = config.min_confidence
        
    def analyze_frame(self, frame: Image.Image) -> Dict:
        try:
            response = self.client.image_to_text(image=frame)
            analysis = json.loads(response)
            return self._validate_analysis(analysis)
        except json.JSONDecodeError:
            return self._create_empty_analysis()

    def _validate_analysis(self, analysis: Dict) -> Dict:
        required_sections = ["metadata", "structure", "components"]
        return analysis if all(section in analysis for section in required_sections) \
            else self._create_empty_analysis()

    def _create_empty_analysis(self) -> Dict:
        return {
            "metadata": {"screen_type": "unknown"},
            "structure": [],
            "components": [],
            "text_elements": [],
            "visual_hierarchy": []
        }

# ====================
# Scene Management
# ====================

class SceneManager:
    def __init__(self, vector_store: ImageVectorStore, config: WorkflowCreatorConfig):
        self.vector_store = vector_store
        self.config = config
        self.scenes = []
        self.current_scene = None
        self.initial_scene_id = None

    def detect_scene_change(self, frame: Image.Image) -> bool:
        if not self.current_scene:
            return True
            
        scene_id, similarity = self.vector_store.find_similar(frame)
        return scene_id != self.current_scene['scene_id']

    def create_new_scene(self, frame: Image.Image, analysis: Dict):
        scene_id = f"scene_{len(self.scenes)+1}"
        self.vector_store.add_embedding(frame, scene_id)
        
        self.current_scene = {
            "scene_id": scene_id,
            "screenshot": self._store_screenshot(frame, scene_id),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "analysis": analysis,
            "actions": []
        }
        
        if not self.initial_scene_id:
            self.initial_scene_id = scene_id
            
        self.scenes.append(self.current_scene)

    def finalize_scene(self):
        if self.current_scene:
            self.current_scene['end_time'] = datetime.now().isoformat()

    def check_completion(self, frame: Image.Image) -> bool:
        if not self.initial_scene_id:
            return False
            
        scene_id, similarity = self.vector_store.find_similar(frame)
        return scene_id == self.initial_scene_id and len(self.scenes) > 1

    def _store_screenshot(self, frame: Image.Image, scene_id: str) -> str:
        path = os.path.join(self.config.screenshot_dir, f"{scene_id}.png")
        frame.save(path)
        return path

# ====================
# Video Processing
# ====================

class VideoProcessor:
    def __init__(self, config: WorkflowCreatorConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.video_path)
        self.vector_store = ImageVectorStore(config)
        self.frame_analyzer = FrameAnalyzer(config)
        self.scene_manager = SceneManager(self.vector_store, config)
        self.frame_count = 0

    def process_video(self) -> Dict:
        workflow = {
            "metadata": self._create_metadata(),
            "scenes": [],
            "embeddings_index": "faiss_index.bin",
            "completed": False
        }

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.frame_count % self.config.frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                analysis = self.frame_analyzer.analyze_frame(pil_image)
                
                if self.scene_manager.detect_scene_change(pil_image):
                    self.scene_manager.finalize_scene()
                    self.scene_manager.create_new_scene(pil_image, analysis)
                else:
                    self._update_current_scene()

                if self.scene_manager.check_completion(pil_image):
                    workflow['completed'] = True
                    break

            self.frame_count += 1

        self._finalize_processing(workflow)
        return workflow

    def _update_current_scene(self):
        # Implement action tracking here
        pass

    def _finalize_processing(self, workflow: Dict):
        self.cap.release()
        faiss.write_index(self.vector_store.index, "faiss_index.bin")
        workflow['scenes'] = [s.copy() for s in self.scene_manager.scenes]

    def _create_metadata(self) -> Dict:
        return {
            "created_at": datetime.now().isoformat(),
            "video_source": self.config.video_path,
            "config": vars(self.config)
        }

# ====================
# Workflow Generation
# ====================

class WorkflowGenerator:
    def generate(self, processed_data: Dict) -> Dict:
        return {
            "workflow": {
                "metadata": processed_data["metadata"],
                "scenes": self._process_scenes(processed_data["scenes"]),
                "embeddings": processed_data["embeddings_index"],
                "completion": processed_data["completed"]
            },
            "statistics": {
                "total_scenes": len(processed_data["scenes"]),
                "unique_screens": len({s['scene_id'] for s in processed_data["scenes"]})
            }
        }

    def _process_scenes(self, scenes: List) -> List:
        return [{
            "id": s["scene_id"],
            "screen_type": s["analysis"]["metadata"]["screen_type"],
            "duration": self._calculate_duration(s["start_time"], s["end_time"]),
            "screenshot": s["screenshot"],
            "structure": s["analysis"]["structure"],
            "components": self._format_components(s["analysis"]["components"]),
            "text_elements": s["analysis"]["text_elements"],
            "actions": s["actions"]
        } for s in scenes]

    def _format_components(self, components: List) -> List:
        return [{
            "type": c.get("component_type"),
            "position": c.get("position"),
            "text": c.get("text_content"),
            "properties": c.get("visual_properties"),
            "actions": c.get("associated_actions")
        } for c in components]

    def _calculate_duration(self, start: str, end: str) -> float:
        if not end:
            return 0.0
        start_time = datetime.fromisoformat(start)
        end_time = datetime.fromisoformat(end)
        return (end_time - start_time).total_seconds()

# ====================
# Main Application
# ====================

def main(config_path: str):
    config = WorkflowCreatorConfig(config_path)
    
    processor = VideoProcessor(config)
    processed_data = processor.process_video()
    
    generator = WorkflowGenerator()
    workflow = generator.generate(processed_data)
    
    with open(config.output_path, 'w') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    
    print(f"Workflow created: {config.output_path}")

# Directly assign the config path for notebook environment
config_path = "/Workspace/Users/mrinalini.vettri@fisglobal.com/video_analysis/workflow creation/config.yaml"
main(config_path)

1
%python
%pip install opencv-python
 
import cv2
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from typing import List, Tuple
 
# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
 
# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
 
def detect_scenes(video_path: str, window_size: int = 100, overlap: int = 50, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Detect scene changes in video using a sliding window approach
    Returns list of (start_time, end_time) tuples in seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
   
    prev_hist = None
    scenes = []
    scene_start = 0.0
    frame_count = 0
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_count += 1
       
        # Convert to grayscale and calculate histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
       
        if prev_hist is not None:
            # Calculate histogram difference
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
           
            if diff < threshold / 100:  # Scene change detected
                scenes.append((scene_start, current_time))
                scene_start = current_time
               
        prev_hist = hist
       
        # Sliding window logic
        if frame_count % window_size == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - overlap)
   
    # Add final scene
    scenes.append((scene_start, current_time))
    cap.release()
    return scenes
 
def extract_key_frame(video_path: str, start_time: float, end_time: float) -> Image.Image:
    """Extract middle frame from a scene segment"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
   
    # Calculate middle frame position
    middle_time = (start_time + end_time) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, middle_time * 1000)
   
    ret, frame = cap.read()
    cap.release()
   
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None
 
def generate_scene_description(frame: Image.Image) -> str:
    """Generate text description of a frame using BLIP-2"""
    inputs = processor(
        images=frame,
        return_tensors="pt"
    ).to(device, torch.float16 if device == "cuda" else torch.float32)
   
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return description
 
def process_video(video_path: str) -> List[dict]:
    """Main processing function"""
    print("Detecting scenes...")
    scenes = detect_scenes(video_path)
    print(f"Found {len(scenes)} scenes")
   
    summaries = []
   
    for i, (start, end) in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}")
       
        # Extract key frame
        frame = extract_key_frame(video_path, start, end)
        if frame is None:
            continue
           
        # Generate description
        description = generate_scene_description(frame)
       
        summaries.append({
            "scene": i+1,
            "start_time": start,
            "end_time": end,
            "duration": end - start,
            "summary": description
        })
   
    return summaries
 
def print_summary(summaries: List[dict]):
    """Print formatted summary"""
    for entry in summaries:
        print(f"\nScene {entry['scene']} ({entry['duration']:.1f}s)")
        print(f"From {entry['start_time']:.1f}s to {entry['end_time']:.1f}s")
        print(f"Summary: {entry['summary']}")
 
if __name__ == "__main__":
    video_path = "/Workspace/Users/ranadhir.ghosh@fisglobal.com/video_analysis/input_video.mp4"  # Change this to your video path
   
    print("Starting video analysis...")
    summaries = process_video(video_path)
   
    print("\nFinal Video Summary:")
    print_summary(summaries)
 
2
Use the following code for traiing purpose : %python
%pip install scenedetect opencv-python qwen-vl-utils peft accelerate datasets
 
import os
from PIL import Image
import cv2
import torch
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
 
# --- Read Hugging Face API Key ---
HF_API_KEY = dbutils.secrets.get(scope="my_secret_scope", key="huggingface_api_key")
os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY
 
# --- Configuration for Fine-Tuning ---
TRAINING_CONFIG = {
    "model_id": "Qwen/Qwen2-VL-7B-Instruct",
    "dataset_path": "/path/to/your/word_actions_dataset",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_steps": 1000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "output_dir": "./word_action_model"
}
 
# --- Initialize Base Model ---
model = Qwen2VLForConditionalGeneration.from_pretrained(
    TRAINING_CONFIG["model_id"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=HF_API_KEY
)
processor = AutoProcessor.from_pretrained(TRAINING_CONFIG["model_id"], use_auth_token=HF_API_KEY)
 
# --- Prepare PEFT/LoRA Configuration ---
peft_config = LoraConfig(
    r=TRAINING_CONFIG["lora_r"],
    lora_alpha=TRAINING_CONFIG["lora_alpha"],
    target_modules=TRAINING_CONFIG["lora_target_modules"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
 
# --- Custom Dataset Preparation ---
def prepare_training_dataset(dataset_path):
    # Sample dataset format:
    # [
    #   {
    #     "image": "mail_merge_step1.jpg",
    #     "description": "User clicks Mailings tab > Start Mail Merge > Step-by-Step Mail Merge Wizard"
    #   },
    #   ...
    # ]
   
    with open(os.path.join(dataset_path, "annotations.json")) as f:
        annotations = json.load(f)
   
    def gen():
        for item in annotations:
            image = Image.open(os.path.join(dataset_path, item["image"]))
            yield {
                "pixel_values": processor(images=image, return_tensors="pt").pixel_values[0],
                "labels": processor(text=item["description"], return_tensors="pt").input_ids[0]
            }
   
    return Dataset.from_generator(gen)
 
# --- Training Function ---
def fine_tune_word_actions():
    dataset = prepare_training_dataset(TRAINING_CONFIG["dataset_path"])
   
    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        max_steps=TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=True,
        logging_steps=10,
        remove_unused_columns=False
    )
   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator
    )
   
    trainer.train()
    model.save_pretrained(TRAINING_CONFIG["output_dir"])
 
# --- Modified Analysis Function with Fine-Tuned Model ---
def analyze_word_action(video_path, scene_timestamps):
    key_frames = extract_key_frames(video_path, scene_timestamps)
   
    # Load fine-tuned model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        TRAINING_CONFIG["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_API_KEY
    )
    model = PeftModel.from_pretrained(model, TRAINING_CONFIG["output_dir"])
   
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": key_frames[0]},
            {"type": "text", "text": "Identify the Microsoft Word action sequence shown:"}
        ]
    }]
   
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[key_frames[0]],
        return_tensors="pt"
    ).to("cuda")
   
    generated_ids = model.generate(**inputs, max_new_tokens=150)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
 
# --- Integration with Existing Pipeline ---
def process_video(video_path):
    scenes = detect_scenes(video_path)
    structured_output = []
   
    for scene in scenes:
        action_sequence = analyze_word_action(video_path, [scene])
        structured_data = {
            "application": "Microsoft Word",
            "detected_action": action_sequence,
            "time_range": f"{scene[0]}s-{scene[1]}s",
            "steps": extract_action_steps(action_sequence)
        }
        structured_output.append(structured_data)
   
    return structured_output
 
# --- Helper Function ---
def extract_action_steps(description):
    prompt = f"""Extract ordered steps from: {description}"""
    inputs = mistral_tokenizer(prompt, return_tensors="pt")
    outputs = mistral_model.generate(**inputs, max_new_tokens=200)
    return mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
 

%python
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
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import json

# --- Read Hugging Face API Key ---
HF_API_KEY = dbutils.secrets.get(scope="my_secret_scope", key="huggingface_api_key")
os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY

# --- Configuration for Fine-Tuning ---
TRAINING_CONFIG = {
    "model_id": "Qwen/Qwen2-VL-7B-Instruct",
    "dataset_path": "/Workspace/Users/mrinalini.vettri@fisglobal.com/video_analysis/sonar_embedding/Training/images",
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
    annotations = [
        {"image": "img1-opening word.png", "description": "Opening a Word file via File > Open menu"},
        {"image": "img2-click on insert menu.png", "description": "Clicking on the Insert menu via the top navigation bar"},
        {"image": "img3-insert shapes.png", "description": "Inserting shapes via Insert > Shapes"},
        {"image": "img4-select shape.png", "description": "Selecting a shape from the Shapes gallery"},
        {"image": "img5-added rectangle shape.png", "description": "Added a rectangle shape via Insert > Shapes > Rectangle"},
        {"image": "img6-select 'add text'.png", "description": "Selecting 'Add Text' via right-click menu on the shape"},
        {"image": "img7-add text to shape.png", "description": "Adding text to the shape via right-click menu > Add Text"},
        {"image": "img8-select connector.png", "description": "Selecting a connector via Insert > Shapes > Connector"},
        {"image": "img9-add connector to flowchart.png", "description": "Adding a connector to the flowchart via Insert > Shapes > Connector"},
        {"image": "img10-change color of shape.png", "description": "Changing the color of the shape via Format > Shape Fill"},
        {"image": "img11-change color of text.png", "description": "Changing the color of the text via Format > Text Fill"},
        {"image": "img12-insert tab.png", "description": "Clicking on the Insert tab via the top navigation bar"},
        {"image": "img13-select header.png", "description": "Selecting a header via Insert > Header"},
        {"image": "img14-added a header.png", "description": "Added a header via Insert > Header > Blank"},
        {"image": "img15-changed title of header.png", "description": "Changed the title of the header via Header & Footer Tools > Design > Title"},
        {"image": "img16-clicked mailings tab.png", "description": "Clicked on the Mailings tab via the top navigation bar"},
        {"image": "img17-start mail merge.png", "description": "Starting a mail merge via Mailings > Start Mail Merge"},
        {"image": "img18-step by step mail merge.png", "description": "Step by step mail merge via Mailings > Start Mail Merge > Step by Step Mail Merge Wizard"},
        {"image": "img19-select recipients.png", "description": "Selecting recipients via Mailings > Select Recipients"},
        {"image": "img20-new address list.png", "description": "Creating a new address list via Mailings > Select Recipients > Type a New List"},
        {"image": "img21-add contents to the list.png", "description": "Adding contents to the list via Mailings > Select Recipients > Type a New List > New Address List"}
    ]

    def gen():
        for item in annotations:
            image_path = os.path.join(dataset_path, item["image"])
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            image = Image.open(image_path)
            try:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
                labels = processor(text=item["description"], return_tensors="pt").input_ids[0]
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
            yield {
                "pixel_values": pixel_values,
                "labels": labels
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

# --- Main Execution ---
if __name__ == "__main__":
    fine_tune_word_actions()
    video_path = "/Workspace/Users/mrinalini.vettri@fisglobal.com/video_analysis/sonar_embedding/Training/workflow videos/video1.mp4"  # Change this to your video path

    print("Starting video analysis...")
    structured_output = process_video(video_path)

    print("\nFinal Video Summary:")
    for entry in structured_output:
        print(f"Application: {entry['application']}")
        print(f"Detected Action: {entry['detected_action']}")
        print(f"Time Range: {entry['time_range']}")
        print(f"Steps: {entry['steps']}")

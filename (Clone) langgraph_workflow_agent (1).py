# Databricks notebook source
'''
%pip install torch torchvision transformers datasets opencv-python faiss-cpu av==14.4.0 numpy==1.26.4

%pip install pillow qwen_vl_utils av==14.4.0

%pip install langgraph
'''

# COMMAND ----------

# DBTITLE 1,Importing required libraries
from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2VLConfig
# Restart the Python kernel to ensure the packages are loaded correctly
#dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,State Schema for Workflow Pipeline Agent
%pip install langgraph
from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated
import json
from langgraph.graph import StateGraph, END

# Define state schema
class InvoiceState(TypedDict):
    """
    Represents the state of our Invoice Processing Agent.
    """
    input_path: str  # Path to the input folder
    invoice_data: Optional[Dict[str, Any]]  # Extracted invoice data
    total_invoices_processed: Optional[int]  # Total number of invoices processed
    average_time_per_invoice: Optional[float]  # Average time per invoice
    current_step: Optional[str] = None  # Current step in the workflow
    error: Optional[str] = None  # Any error message from execution
    final_response: Optional[Dict[str, Any]] = None  # To get final results

# Create widgets for user input
dbutils.widgets.text("input_path", "", "Input Path (File or Folder)*")
dbutils.widgets.dropdown("extract_all", "true", ["true", "false"], "Extract all sections")
dbutils.widgets.dropdown("extract_invoice_amount", "true", ["true", "false"], "Extract invoice amount")
dbutils.widgets.dropdown("extract_itemise", "true", ["true", "false"], "Extract itemised sections")

# Get user input
input_path = dbutils.widgets.get("input_path")
extract_all = dbutils.widgets.get("extract_all") == "true"
extract_invoice_amount = dbutils.widgets.get("extract_invoice_amount") == "true"
extract_itemise = dbutils.widgets.get("extract_itemise") == "true"

# COMMAND ----------

# DBTITLE 1,user_input node
from typing import TypedDict, Optional, Dict, Any
from typing_extensions import Annotated
import json
from langgraph.graph import StateGraph, END

# Define state schema
class WorkflowAgentState(TypedDict):
    """
    Represents the state of our Workflow Processing Agent.
    """
    input_path: str  # Path to the input folder
    extract_all: bool  # Whether to extract all sections
    extract_invoice_amount: bool  # Whether to extract invoice amount
    extract_itemise: bool  # Whether to extract itemised sections
    invoice_data: Optional[Dict[str, Any]]  # Extracted invoice data
    total_invoices_processed: Optional[int]  # Total number of invoices processed
    average_time_per_invoice: Optional[float]  # Average time per invoice
    current_step: Optional[str] = None  # Current step in the workflow
    error: Optional[str] = None  # Any error message from execution
    final_response: Optional[Dict[str, Any]] = None  # To get final results

# Create widgets for user input
dbutils.widgets.text("input_path", "", "Input Path (File or Folder)*")
dbutils.widgets.dropdown("extract_all", "true", ["true", "false"], "Extract all sections")
dbutils.widgets.dropdown("extract_invoice_amount", "true", ["true", "false"], "Extract invoice amount")
dbutils.widgets.dropdown("extract_itemise", "true", ["true", "false"], "Extract itemised sections")

# Define the user input node function
def user_input_node(state: WorkflowAgentState) -> WorkflowAgentState:
    input_path = dbutils.widgets.get("input_path")
    if not input_path:
        raise ValueError("Input path is required.")
    
    extract_all = dbutils.widgets.get("extract_all") == "true"
    extract_invoice_amount = dbutils.widgets.get("extract_invoice_amount") == "true"
    extract_itemise = dbutils.widgets.get("extract_itemise") == "true"
    
    state["input_path"] = input_path
    state["extract_all"] = extract_all
    state["extract_invoice_amount"] = extract_invoice_amount
    state["extract_itemise"] = extract_itemise
    return state

# COMMAND ----------

# DBTITLE 1,InvoiceAnalyzer
import json
from typing import Dict, List, Optional
from PIL import Image
import os
import time
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2VLConfig
 
class InvoiceAnalyzer:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.prompt_template = '''You are an image analysis expert and an invoice image is provided to you. Your task is to extract all information from the image and organize it into a structured JSON format.
 
    The extracted data should be categorized into:
 
    1. Tabular Data : A table is defined as a grid-like arrangement of data organized into rows and columns, often with headers. Ignore any surrounding text, logos, signatures, or decorative elements when extracting tabular data. It typically includes line items such as Quantity, Description, Unit Price, Discount and Amount.
 
    2. Non-Tabular Text : Non-tabular text refers to any content that is not organized in a grid-like structure of rows and columns. This includes paragraphs, headers, footers, labels, annotations, and any standalone text blocks. Ignore tables, charts, and graphical elements. This tipically includes metadata such as Invoice Number, Invoice Date, PO Number, Due Date, Billing Addresses and Shipping Addresses, Company Information, Tax Rate, Calculated Tax Amounts, Discount rate, Total Discount Value, Subtotal, Total, and Payment Terms.
 
    3. Non_table_image Elements : Any other elements such as Signatures, lLgos, or Stamps, described with associated text. Simply ignore this and exclude from JSON.
   
    Important Instructions:
    Use your understanding of invoice structure to group non-tabular data into meaningful sections such as:
      -"Invoice Details" – e.g. invoice number, date, PO number, due date
      -"Billing Info" – e.g. bill to and ship to names and addresses
      -"Organization Info"
      -"Financial Summary" – e.g., subtotal, tax rate, tax amount, GST rate, GST Amount,  total amount due
      -"Terms and Conditions" – e.g., payment terms or due in days
   
    # If any rate%, calculated tax, subtotal, or total is found, include it under "Financial Summary" as key-value pairs.
 
    # Insome cases the field names and corresponding values are provided side by side, in other cases the field names are not provided and the values are scattered across the image. In such cases, use your understanding of invoice structure to group the values into meaningful sections and make sure no information is missed.
 
    # In some cases the field names are not provided rather a total is given below some table columms so make sure this this information also comes under Finacial Summary as key-value pairs like Total Payment Amount, Total Discount etc .  
 
    # The company name and address are found under 'From'.
   
    # Do not try to read the company logo or any other image, skip if not found.
   
    # You can omit a section from sample JSON below if it is not present in the image.
 
    # Make it double sure that no data is left in the image without being added to the JSON.
 
    Ensure the output is valid JSON and follows this structure:                          
    {
      "invoice_data": {
        "table": {
          "items_table": {
            "headers": ["Key1", "Key2", "Key3", "Key4"],
            "rows": [
                {
                    "key1": "value1",
                    "key2": "value2",
                    "key3": "value3",
                    "key4": "value4"
                },
                {...}]
          }
        },
        "non_table_text": {
          "section_name": { "key": "...", "data": { ... } },
          "section_2": { "group_name": "...", "data": { ... } },
          ...
        }               
      }
    }
    '''
 
    def extract_all(self, json_data: Dict) -> Dict:
        result = {
            "items_table": json_data.get("invoice_data", {}).get("table", {}).get("items_table", {}),
            "non_table_text": json_data.get("invoice_data", {}).get("non_table_text", {}),
            "non_table_image": json_data.get("invoice_data", {}).get("non_table_image", {})
        }
        return result
 
    def extract_itemize(self, json_data: Dict) -> Dict:
        result = {
            "items_table": json_data.get("invoice_data", {}).get("table", {}).get("items_table", {})
        }
        return result
 
    def extract_invoice_amount(self, json_data: Dict) -> Optional[float]:
        non_table_text_data = json_data.get("invoice_data", {}).get("non_table_text", {})
        amount_keys = ["Sum Total", "Invoice Amount", "Amount", "Total"]
        amount = None
       
        for section in non_table_text_data.values():
            if isinstance(section, dict):
                for key in amount_keys:
                    if key in section.get("data", {}):
                        amount_str = section["data"][key]
                        amount = float(amount_str.replace(",", ""))
                        break
            if amount is not None:
                break
       
        return amount
 
# Initialize Qwen2-VL model and processor (only once)
model = None
processor = None
 
def initialize_model():
    global model, processor
    if model is None or processor is None:
        print("Initializing model...")
        local_model_path = "/dbfs/FileStore/model/Qwen/Qwen2-VL-7B-Instruct"
       
        try:
            # Load the configuration
            print("Loading configuration...")
            config = Qwen2VLConfig.from_pretrained(local_model_path)
            print("Configuration loaded.")
           
            # Initialize the model with the configuration
            print("Loading model...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            print("Model loaded successfully.")
           
            # Initialize the processor
            print("Loading processor...")
            processor = AutoProcessor.from_pretrained(local_model_path)
            print("Processor loaded successfully.")
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            return None, None
       
        return model, processor
 
# Ensure the model is initialized
model, processor = initialize_model()
 
# Check if model and processor are initialized successfully
if model is None or processor is None:
    raise RuntimeError("Failed to initialize model and processor.")
 
def analyze_scene_using_Qwen(images):
    analyzer = InvoiceAnalyzer("")  # Create an instance of InvoiceAnalyzer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": analyzer.prompt_template}  # Access prompt_template through the instance
            ],
        }
        for image in images
    ]
    texts = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs = images
 
    # Move only required tensors to GPU, keep others on CPU
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.cuda(non_blocking=True) if hasattr(v, "cuda") else v for k, v in inputs.items()}
 
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
 
    return output_texts
 
def process_image_batch(image_paths):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Resize the image
            new_size = (int(img.width * 0.5), int(img.height * 0.5))
            resized_image = img.resize(new_size, Image.LANCZOS)
            images.append(resized_image.convert("RGB"))
    descriptions = analyze_scene_using_Qwen(images)
    results = []
    for image_path, description in zip(image_paths, descriptions):
        try:
            json_data = json.loads(description)
        except Exception:
            json_data = description
        results.append((os.path.splitext(os.path.basename(image_path))[0], json_data))
    return results
 
def save_json(output_folder, base_name, json_data):
    output_path = os.path.join(output_folder, f"{base_name}.json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
 
def process_images(image_folder, output_folder, batch_size=2, num_images=None):
    print("Processing images in batches ...")
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
   
    if num_images is not None:
        image_paths = image_paths[:num_images]
   
    total_images = len(image_paths)
    total_time = 0
   
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        start_time = time.time()
        batch_results = process_image_batch(batch_paths)
        batch_time = time.time() - start_time
        total_time += batch_time
        print(f"Batch {i//batch_size + 1} processed in {batch_time:.2f} seconds")
       
        # Write individual JSON files
        for base_name, json_data in batch_results:
            save_json(output_folder, base_name, json_data)
   
    # Calculate average processing time per image
    avg_time_per_image = total_time / total_images
    stats = {
        "total_images_processed": total_images,
        "total_processing_time": total_time,
        "average_processing_time_per_image": avg_time_per_image
    }
   
    # Write stats to a separate JSON file
    stats_output_path = os.path.join(output_folder, "processing_stats.json")
    with open(stats_output_path, 'w') as f:
        json.dump(stats, f, indent=4)
   
    print(f"Average processing time per image: {avg_time_per_image:.2f} seconds")
 
# Example usage
image_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data"
output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
batch_size = 5  # Specify the batch size (no of images in one batch)
num_images = 10  # Specify the number of images to process
process_images(image_folder, output_folder, batch_size, num_images)

# COMMAND ----------

# DBTITLE 1,extract_all node
import os
import json

def extract_all_node(state: dict) -> dict:
    import time
    start_time = time.time()
    
    analyzer = InvoiceAnalyzer(state["input_path"])
    
    # Process images in real-time
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5  # Specify the batch size (no of images in one batch)
    num_images = 10  # Specify the number of images to process
    
    process_images(image_folder, output_folder, batch_size, num_images)
    
    # Collect the node-related information from the generated JSON files
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json') and fname != "processing_stats.json"]
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file) if isinstance(json.load(file), dict) else json.loads(json.load(file))  # Parse the JSON string into a dictionary if needed
            all_data.append(analyzer.extract_all(json_data))
    
    state["invoice_data"] = all_data
    
    # Calculate total invoices processed and average time per invoice
    state["total_invoices_processed"] = len(json_files)
    state["average_time_per_invoice"] = (time.time() - start_time) / len(json_files)
    
    return state

# Example usage
state = {
    "input_path": "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data"
}
state = extract_all_node(state)
display(state)

# COMMAND ----------

# DBTITLE 1,extract_invoice_amount node
def extract_invoice_amount_node(state: InvoiceState) -> InvoiceState:
    import time
    start_time = time.time()
    
    analyzer = InvoiceAnalyzer(state["input_path"])
    
    # Process images in real-time
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5  # Specify the batch size (no of images in one batch)
    num_images = 10  # Specify the number of images to process
    
    process_images(image_folder, output_folder, batch_size, num_images)
    
    # Collect the node-related information from the generated JSON files
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json') and fname != "processing_stats.json"]
    invoice_amounts = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            amount = analyzer.extract_invoice_amount(json_data)
            if amount is not None:
                invoice_amounts.append(amount)
    
    state["invoice_amount"] = invoice_amounts
    
    # Calculate total invoices processed and average time per invoice
    state["total_invoices_processed"] = len(json_files)
    state["average_time_per_invoice"] = (time.time() - start_time) / len(json_files)
    
    return state

# Example usage
state = InvoiceState(input_path="/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data")
state = extract_invoice_amount_node(state)
display(state)

# COMMAND ----------

# DBTITLE 1,extract_itemize_node (Includes all the items in the table)
def extract_itemize_node(state: InvoiceState) -> InvoiceState:
    import time
    start_time = time.time()
    
    analyzer = InvoiceAnalyzer(state["input_path"])
    
    # Process images in real-time
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5  # Specify the batch size (no of images in one batch)
    num_images = 10  # Specify the number of images to process
    
    process_images(image_folder, output_folder, batch_size, num_images)
    
    # Collect the node-related information from the generated JSON files
    json_files = [os.path.join(output_folder, fname) for fname in os.listdir(output_folder) if fname.endswith('.json') and fname != "processing_stats.json"]
    itemized_data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
            itemized_data.append(analyzer.extract_itemize(json_data))
    
    state["items_table"] = itemized_data
    
    # Calculate total invoices processed and average time per invoice
    state["total_invoices_processed"] = len(json_files)
    state["average_time_per_invoice"] = (time.time() - start_time) / len(json_files)
    
    return state

# Example usage
state = InvoiceState(input_path="/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data")
state = extract_itemize_node(state)
display(state)

# COMMAND ----------

# DBTITLE 1,result_node
def result_node(state: InvoiceState) -> InvoiceState:
    # Call the process_images function
    image_folder = state["input_path"]
    output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
    batch_size = 5  # Specify the batch size (no of images in one batch)
    num_images = 10  # Specify the number of images to process
    
    process_images(image_folder, output_folder, batch_size, num_images)
    
    # Read the stats from the generated JSON file
    stats_output_path = os.path.join(output_folder, "processing_stats.json")
    with open(stats_output_path, 'r') as f:
        stats = json.load(f)
    
    final_response = {
        "total_invoices_processed": stats["total_images_processed"],
        "average_time_per_invoice": f"{stats['average_processing_time_per_image']:.2f} seconds"
    }
    state['final_response'] = final_response
    return state

# Example usage
state = InvoiceState(input_path="/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/amazon_data")
state = result_node(state)
display(state)

# COMMAND ----------

# DBTITLE 1,langgraph workflow
from langgraph.graph import StateGraph, END
import os
import json

# Define the graph
workflow = StateGraph(dict)

# Add all nodes
workflow.add_node("user_input_node", user_input_node)
workflow.add_node("extract_all_node", extract_all_node)
workflow.add_node("extract_invoice_amount_node", extract_invoice_amount_node)
workflow.add_node("extract_itemize_node", extract_itemize_node)
workflow.add_node("result_node", result_node)

# Set entry point
workflow.set_entry_point("user_input_node")

# Conditional branching logic
def route_based_on_user_selection(state: dict) -> str:
    if state.get("extract_all"):
        return "extract_all_node"
    elif state.get("extract_invoice_amount"):
        return "extract_invoice_amount_node"
    elif state.get("extract_itemise"):
        return "extract_itemize_node"
    else:
        return "result_node"  # fallback

# Add router node
workflow.add_conditional_edges(
    "user_input_node",
    route_based_on_user_selection,
    {
        "extract_all_node": "extract_all_node",
        "extract_invoice_amount_node": "extract_invoice_amount_node",
        "extract_itemize_node": "extract_itemize_node",
        "result_node": "result_node"
    }
)

# Connect each processing node to result node
workflow.add_edge("extract_all_node", "result_node")
workflow.add_edge("extract_invoice_amount_node", "result_node")
workflow.add_edge("extract_itemize_node", "result_node")

# End the workflow
workflow.add_edge("result_node", END)

# Compile the graph
invoice_workflow = workflow.compile()

# Execute the workflow
initial_state = {
    "input_path": input_path,
    "extract_all": extract_all,
    "extract_invoice_amount": extract_invoice_amount,
    "extract_itemise": extract_itemise
}
final_state = invoice_workflow.invoke(initial_state)

# Display final results
result_data = {
    "processed_count": final_state['final_response']['total_invoices_processed'],
    "average_processing_time_per_invoice": final_state['final_response']['average_time_per_invoice'],
    "is_folder": os.path.isdir(final_state['input_path'])
}

# Exit notebook with results
dbutils.notebook.exit(json.dumps({
    "status": "success",
    "result": result_data
}))

# COMMAND ----------

# DBTITLE 1,generating the final result data
result_data = {
    "processed_count": state['final_response']['total_invoices_processed'],
    "average_processing_time_per_invoice": state['final_response']['average_time_per_invoice'],
    "is_folder": os.path.isdir(state['input_path'])
}

# Exit the notebook with the result data
dbutils.notebook.exit(json.dumps({
    "status": "success",
    "result": result_data
}))
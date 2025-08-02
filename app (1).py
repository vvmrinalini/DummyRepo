%python
import os
import json
import logging
import random
from flask import Flask, request, render_template, jsonify, send_from_directory
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState
import pandas as pd

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Initialize Flask app
flask_app = Flask(__name__)

img_folder = os.path.join("static", "image")
flask_app.config["UPLOAD_FOLDER"] = img_folder

databricks = WorkspaceClient()

def read_processing_stats(output_folder):
    stats_output_path = os.path.join(output_folder, "processing_stats.json")
    if os.path.exists(stats_output_path):
        with open(stats_output_path, 'r') as f:
            return json.load(f)
    return {}

def update_processing_stats(output_folder, stats):
    stats_output_path = os.path.join(output_folder, "processing_stats.json")
    with open(stats_output_path, 'w') as f:
        json.dump(stats, f, indent=4)

@flask_app.route('/', methods=['GET', 'POST'])
def index():
    send_img = os.path.join(flask_app.config["UPLOAD_FOLDER"], "send-message.png")
    return render_template("index.html", send_img=send_img)

@flask_app.route('/invoice_run', methods=['POST'])
def invoice_run():
    try:
        req_data = request.get_json()
        input_path = req_data.get('input_path')
        extract_all = req_data.get('extract_all', True)
        extract_invoice_amount = req_data.get('extract_invoice_amount', False)
        extract_itemise = req_data.get('extract_itemise', False)

        if not input_path:
            return jsonify({"status": "Error", "error": "Input path is required"}), 400

        if not input_path.startswith("/dbfs/"):
            input_path = f"/dbfs/FileStore/invoicesense/{input_path}"

        is_folder = not any(input_path.lower().endswith(ext) for ext in ['.tif', '.jpg', '.png', '.jpeg'])

        # Randomly select images from DBFS if input_path is a folder
        if is_folder:
            files = list(databricks.dbfs.list(input_path))
            image_files = [f.path for f in files if f.path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg'))]
            if not image_files:
                return jsonify({"status": "Error", "error": "No images found in the specified folder"}), 400
            input_path = random.choice(image_files)

        run_id = databricks.jobs.run_now(
            job_id=899879803593620,  # Replace with your actual job ID for invoice processing
            notebook_params={
                "input_path": input_path,
                "extract_all": str(extract_all).lower(),
                "extract_invoice_amount": str(extract_invoice_amount).lower(),
                "extract_itemise": str(extract_itemise).lower()
            }
        ).run_id

        return jsonify({"run_id": run_id})

    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

@flask_app.route('/check_invoice_status', methods=['POST'])
def check_invoice_status():
    data = request.get_json()
    run_id = data.get('run_id')
    status = ""
    result = ""

    try:
        run_status = databricks.jobs.get_run(run_id)
        
        if run_status.state.life_cycle_state == RunLifeCycleState.TERMINATED:
            if run_status.state.result_state == RunResultState.SUCCESS:
                notebook_output = databricks.jobs.get_run_output(run_status.tasks[0].run_id).notebook_output
                
                if notebook_output.result:
                    try:
                        result = json.loads(notebook_output.result)
                        if isinstance(result, dict) and "results" in result:
                            output_folder = "/Workspace/Users/mrinalini.vettri@fisglobal.com/Invoice Sense/OutputData/json_test"
                            stats = read_processing_stats(output_folder)
                            return jsonify({
                                "status": "Succeed",
                                "result": result,
                                "is_folder": True,
                                "stats": stats
                            })
                    except json.JSONDecodeError:
                        result = {"raw_output": notebook_output.result}
                
                return jsonify({
                    "status": "Succeed",
                    "result": result,
                    "is_folder": False
                })
            else:
                result = f"Job failed: {run_status.state.state_message}"
                status = "Not-Succeed"
        else:
            result = f"Job status: {run_status.state.life_cycle_state.value}"
            status = "Not-Finish"
        
        return jsonify({
            "status": status,
            "result": result
        })
   
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)})

@flask_app.route('/list_images', methods=['GET'])
def list_images():
    try:
        files = list(databricks.dbfs.list("/dbfs/FileStore/invoicesense"))
        image_files = [f.path.split('/')[-1] for f in files if f.path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg'))]
        folders = [f.path.split('/')[-1] for f in files if f.is_dir]
        return jsonify({"status": "success", "images": image_files, "folders": folders}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@flask_app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory('/dbfs/dbfs/FileStore/invoicesense', filename)

@flask_app.route('/agent_run', methods=['POST'])
def agent_run():
    try:
        req_data = request.get_json()
        prompt = req_data.get('prompt')
        if not prompt:
            return jsonify({"status": "Error", "error": "Prompt is required"}), 400
        run_id = databricks.jobs.run_now(
            job_id=97013954614133,  # Replace with your actual job ID for the agent
            notebook_params={"user_question": prompt}
        ).run_id
        return jsonify({
            "status": "Success",
            "message": "Agent job submitted.",
            "run_id": run_id
        })
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

@flask_app.route('/check_agent_status', methods=['POST'])
def check_agent_status():
    data = request.get_json()
    run_id = data.get('run_id')

    try:
        run_status = databricks.jobs.get_run(run_id)

        if run_status.state.life_cycle_state == RunLifeCycleState.TERMINATED:
            if run_status.state.result_state == RunResultState.SUCCESS:
                task_runs = run_status.tasks
                if not task_runs:
                    return jsonify({"status": "FAILED", "error": "No task runs found."})

                notebook_task_run_id = task_runs[0].run_id
                output = databricks.jobs.get_run_output(notebook_task_run_id)

                result = output.notebook_output.result if output.notebook_output else None

                if not result:
                    return jsonify({"status": "FAILED", "error": "No result returned."})

                return jsonify({
                    "status": "SUCCEEDED",
                    "result": result
                })

            else:
                return jsonify({
                    "status": "FAILED",
                    "error": run_status.state.state_message
                })

        else:
            return jsonify({
                "status": run_status.state.life_cycle_state.value
            })

    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)})

def submit_agent_job(prompt):
    with flask_app.test_request_context():
        response = flask_app.test_client().post(
            '/agent_run',
            json={"prompt": prompt}
        )
        return response.get_json()

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=False)
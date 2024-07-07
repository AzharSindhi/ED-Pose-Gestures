import json
import mlflow
import os
from tqdm import tqdm
from datetime import datetime
import glob
import multiprocessing

# Function to read JSON data from the file
exclude_dirs = [
    "gestures",
    "gestures_allclasses_coco_pretrained_r50",
    "gestures_classifier_multiple_hpc",
    "gestures_classifier_multiple_lme",
    "gestures_classifier1",
    "gestures_persononly_coco_pretrained_r50",
    "humanart_debug",
    "humanart_r50_gestures_persononly_old",
    "classi_token", 
    # "edpose",
    # "extratoken",
    "classifier2",
    "classifier1",
    "edpose_nd8",
    "edpose_original"
]
key_mapping = {
    "test_coco_eval_bbox": "testAP"
}
mlflow.set_tracking_uri(uri="http://127.0.0.1:5002")   

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_metrics(path):
    assert os.path.exists(path), f"Metrics path {path} not found"
    data = []
    with open(path, 'r') as file:
        lines = file.read().strip().splitlines()
        for line in lines:
            line_data = json.loads(line)
            data.append(line_data)
    return data

def log_hyperparams(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} not found")
        return
    params_dict = read_json_file(file_path)
    mlflow.log_params(params_dict) 

# Function to log metrics using MLflow
def log_metrics_with_mlflow(metrics):
    # Log metrics
    for key, value in metrics.items():
        if "time" in key:
            continue
        key = key.replace("(", "_")
        key = key.replace(")", "")
        if isinstance(value, list):
            key = key.split("_")[0]
            mlflow.log_metric(f"{key}AP", float(value[0]))
            # for i, item in enumerate(value):
            #     mlflow.log_metric(key=f"{key}_{i}", value=item)
        else:
            try:
                value = float(value)
            except Exception as e:
                print(f"{key} is not a number")
            
            mlflow.log_metric(key=key, value=value)

def log_artificats(dir_path):
    val_coco_best = os.path.join(dir_path, "val_coco_results_best.txt")
    val_coco_last = os.path.join(dir_path, "val_coco_results_last.txt")
    if os.path.exists(val_coco_best):
        mlflow.log_artifact(val_coco_best)
    if os.path.exists(val_coco_last):
        mlflow.log_artifact(val_coco_last)

def log_run(dir_path, experiment_name, run_name):
    run_name = experiment_name + "-" + run_name.split("_")[0] + "-" + run_name.split("_")[-1]
    model_type = dir_path.split("/")[-3]
    mlflow.set_experiment("sensoryArt_runs")
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"{model_type}_{run_name}")
        print("started mlflow")
        # Path to the text file containing the JSON data
        params_path = os.path.join(dir_path, "config_args_raw.json")
        metrics_path = os.path.join(dir_path, "log.txt")
        
        # Read JSON data from the file
        print("logging metrics")
        log_hyperparams(params_path)
        metrics_epoch_data = read_metrics(metrics_path)
        # Log metrics using MLflow
        for metrics_data in metrics_epoch_data:
            log_metrics_with_mlflow(metrics_data)

        log_artificats(dir_path)


def log_all_info(log_dir):
    experiment_names = os.listdir(log_dir)
    for experiment_name in experiment_names:
        run_names = os.listdir(os.path.join(log_dir, experiment_name))
        for run_name in run_names:
            run_dir = os.path.join(log_dir, experiment_name, run_name)
            log_run(run_dir, experiment_name, run_name)
 
if __name__ == "__main__":
        # Start MLflow run
    base_dir = "logs/train_sensory_selected_new"
    all_logs = os.listdir(base_dir)

    dirs_to_log = all_logs #[logdir for logdir in all_logs if logdir not in exclude_dirs]
    print(dirs_to_log)

    for dir in tqdm(dirs_to_log):
        log_dir = os.path.join(base_dir, dir)
        log_all_info(log_dir)



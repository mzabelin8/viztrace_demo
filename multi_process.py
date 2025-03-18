from viztracer import VizTracer
import multiprocessing as mp
import numpy as np
import time
from sklearn.datasets import make_classification, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

def init_viztracer(process_id):
    tracer = VizTracer(
        output_file=f"viztracer_process_{process_id}.json",
        log_gc=True
    )
    tracer.start()
    return tracer

def train_specific_model(process_id, model_type, X, y, queue):
    tracer = init_viztracer(process_id)
    
    pid = os.getpid()
    
    if model_type != "RandomForest":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    if model_type == "LogisticRegression":
        time.sleep(0.5)
        model = LogisticRegression(max_iter=300)
    elif model_type == "SVM":
        time.sleep(1)
        model = SVC(gamma='auto')
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    for _ in range(10):
        np.random.random((500, 500)).dot(np.random.random((500, 500)))
    
    print(f"Process {process_id}: {model_type} завершено с точностью {accuracy:.4f}")
    
    tracer.stop()
    tracer.save()
    
    result = {
        "process_id": process_id,
        "model_type": model_type,
        "accuracy": accuracy,
        "pid": pid
    }
    queue.put(result)

def prepare_dataset(dataset_id):
    time.sleep(0.2)
    
    if dataset_id == 0:
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_classes=2, 
            random_state=42
        )
        dataset_name = "synthetic"
    elif dataset_id == 1:
        digits = load_digits()
        X, y = digits.data, digits.target
        dataset_name = "digits"
    else:
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_classes=3, 
            n_informative=5, 
            random_state=42
        )
        dataset_name = "synthetic_complex"
    
    return X, y, dataset_name

def process_function(process_id, dataset_id, model_type, queue):
    X, y, dataset_name = prepare_dataset(dataset_id)
    
    train_specific_model(process_id, model_type, X, y, queue)
    
    time.sleep(0.3)

def main():
    process_configs = [
        {"id": 0, "dataset": 0, "model": "LogisticRegression"},
        {"id": 1, "dataset": 1, "model": "SVM"},
        {"id": 2, "dataset": 2, "model": "RandomForest"},
    ]
    
    result_queue = mp.Queue()
    
    processes = []
    for config in process_configs:
        process = mp.Process(
            target=process_function,
            args=(
                config["id"],
                config["dataset"],
                config["model"],
                result_queue
            )
        )
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

if __name__ == "__main__":
    main_tracer = VizTracer(
        output_file="viztracer_main_process.json",
        log_gc=True
    )
    
    main_tracer.start()
    
    main()
    
    main_tracer.stop()
    main_tracer.save()

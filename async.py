from viztracer import VizTracer
import asyncio
import time
import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import concurrent.futures

async def fetch_dataset(dataset_id, delay=None):
    if delay is None:
        delay = random.uniform(0.5, 2.0)
    
    await asyncio.sleep(delay)
    
    n_samples = 1000 + dataset_id * 200
    n_features = 10 + dataset_id * 5
    random_state = 42 + dataset_id
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=2,
        random_state=random_state
    )
    
    return {
        "id": dataset_id,
        "X": X,
        "y": y,
        "name": f"synthetic_dataset_{dataset_id}"
    }

async def preprocess_data(dataset, intensity=50):
    dataset_id = dataset["id"]
    X = dataset["X"]
    y = dataset["y"]
    
    result = X.copy()
    chunk_size = intensity
    
    for i in range(0, intensity, chunk_size):
        await asyncio.sleep(0.01)
        
        end = min(i + chunk_size, intensity)
        for _ in range(i, end):
            result = np.sin(result) + np.cos(result**2) * 0.01
    
    return {
        "id": dataset_id,
        "X": result,
        "y": y,
        "name": dataset["name"],
        "preprocessed": True
    }

def train_model_blocking(model_type, X, y, dataset_id):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    if model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=300)
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    training_time = time.time() - start_time
    
    print(f"Обучение {model_type} для датасета {dataset_id} завершено с точностью {accuracy:.4f}")
    
    return {
        "dataset_id": dataset_id,
        "model_type": model_type,
        "accuracy": accuracy,
        "training_time": training_time,
        "model": model
    }

async def train_model(model_type, dataset, executor):
    dataset_id = dataset["id"]
    X = dataset["X"]
    y = dataset["y"]
    
    result = await asyncio.get_event_loop().run_in_executor(
        executor, train_model_blocking, model_type, X, y, dataset_id
    )
    
    return result

async def progress_reporter(total_seconds):
    for i in range(total_seconds * 2):
        await asyncio.sleep(0.5)

async def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    try:
        progress_task = asyncio.create_task(progress_reporter(30))
        
        datasets_tasks = [
            fetch_dataset(0, delay=1.5),
            fetch_dataset(1, delay=0.7),
            fetch_dataset(2, delay=2.0)
        ]
        datasets = await asyncio.gather(*datasets_tasks)
        
        preprocess_tasks = [
            preprocess_data(datasets[0], intensity=30),
            preprocess_data(datasets[1], intensity=60),
            preprocess_data(datasets[2], intensity=40)
        ]
        processed_datasets = await asyncio.gather(*preprocess_tasks)
        
        training_tasks = [
            train_model("LogisticRegression", processed_datasets[0], executor),
            train_model("RandomForest", processed_datasets[0], executor),
            train_model("LogisticRegression", processed_datasets[1], executor),
            train_model("RandomForest", processed_datasets[2], executor),
        ]
        model_results = await asyncio.gather(*training_tasks)
        
        progress_task.cancel()
        
        return model_results
    
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    tracer = VizTracer(
        log_async=True,
        log_gc=True
    )
    
    tracer.start()
    
    results = asyncio.run(main())
    
    tracer.stop()
    tracer.save("viztracer_async_ml.json")

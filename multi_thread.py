from viztracer import VizTracer
import threading
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess_data(X, intensity=100):
    result = X.copy()
    for _ in range(intensity):
        result = np.sin(result) + np.cos(result**2)
    return result

def train_model(thread_id, X, y, intensity=100):
    X_processed = preprocess_data(X, intensity)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Thread {thread_id}: Завершено с точностью {accuracy:.4f}")
    
    return model, accuracy

def thread_function(thread_id):
    X, y = make_classification(
        n_samples=500, 
        n_features=10, 
        n_classes=2, 
        random_state=thread_id
    )
    
    intensity = (thread_id + 1) * 50
    
    time.sleep(thread_id * 0.2)
    
    model, accuracy = train_model(thread_id, X, y, intensity)
    
    time.sleep(0.5)
    
    return model, accuracy

def main():
    threads = []
    for i in range(3):
        thread = threading.Thread(target=thread_function, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    tracer = VizTracer(
        log_gc=True,
        log_async=True
    )
    tracer.start()
    
    main()
    
    tracer.stop()
    tracer.save("viztracer_multithread_ml.json")

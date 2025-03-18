# viztrace_demo



```
pip install -r requirements.txt

```

### Multi Thread Support

```
python multi_thread.py

vizviewer viztracer_multithread_ml.json
```
### Multi Process Support


```
python multi_process.pymulti_thread.py

viztracer --combine viztracer_process_*.json -o viztracer_multiprocess_combined.json

vizviewer viztracer_multiprocess_combined.json
```
### Async Support

```
python async.py

vizviewer viztracer_async_ml.json
```
### Other features

- User event
- Variables
- Logs

```
python example.py

vizviewer viztracer_text_analysis.json
```

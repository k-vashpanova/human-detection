# human-detection
Программа по распознаванию людей на видео при помощи моделей YOLO от Ultralytics.

## Установка
```bash
pip install git+https://github.com/k-vashpanova/human-detection
```
## Использование
CLI
```bash
# Обработка input_video.mp4 моделью по умолчанию (yolo11n.pt)
detect_humans input_video.mp4

# Обработка input_video.mp4 моделью yolo11s.pt и сохранение
# в файл output_video.avi
detect_humans input_video.mp4 -o output_video.avi -m yolo11s.pt
```
Python
```python
>>> from detect_humans import detect_humans_in_video

# Обработка input_video.mp4 моделью по умолчанию (yolo11n.pt)
>>> detect_humans_in_video('input_video.mp4')

Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt': 100% ━━━━━━━━━━━━ 5.4MB 90.9MB/s 0.1s
Processing video: 100% 705/705 [02:17<00:00,  5.14it/s]
Video 'output.avi' created successfully.

# Обработка input_video.mp4 моделью yolo11s.pt и сохранение
# в файл output_video.avi
>>> detect_humans_in_video('input_video.mp4',
                           output_filename='output_video.avi',
                           model_path='yolo11s.pt')
```
Подробнее о поддерживаемых моделях см. на <a href="https://docs.ultralytics.com/models/">сайте Ultralytics</a>.

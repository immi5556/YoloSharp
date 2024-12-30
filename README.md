# YoloSharp

Train Yolo model in C# with TorchSharp. </br>
With the help of this project you won't have to transform .pt model to onnx, and can train your own model in C#.

## Feature

- Written in C# only.
- Train and predict your own model.
- Support Yolov5, Yolov5u, Yolov8 and Yolov11 now.
- Support n/s/m/l/x size.
- Support LetterBox and Mosaic4 method for preprocessing images.
- Support NMS with GPU.
- Support Load PreTrained models from ultralytics/yolov5/yolov8 and yolo11 (converted).


## Models

You can download yolov5/yolov8 pre-trained models here.

| model | n| s | m | l | x |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yolov5 | [yolov5n](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5n.bin) | [yolov5s](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5s.bin) | [yolov5m](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5m.bin) | [yolov5l](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5l.bin) | [yolov5x](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5x.bin) |
| yolov5 | [yolov5nu](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5nu.bin) | [yolov5su](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5su.bin) | [yolov5mu](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5mu.bin) | [yolov5lu](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5lu.bin) | [yolov5xu](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov5xu.bin) |
| yolov8 | [yolov8n](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov8n.bin) | [yolov8s](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov8s.bin) | [yolov8m](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov8m.bin) | [yolov8l](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov8l.bin) | [yolov8x](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov8x.bin) |
| yolov11 | [yolov11n](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov11n.bin) | [yolov11s](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov11s.bin) | [yolov11m](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov11m.bin) | [yolov11l](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov11l.bin) | [yolov11x](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.4/Yolov11x.bin) |

For example:

Use yolov5n pre-trained model to detect coco128

![image](https://github.com/user-attachments/assets/d32f7805-9f98-4530-bda6-43630c765159)


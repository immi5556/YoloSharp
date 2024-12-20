# YoloSharp

Train Yolo model in C# with TorchSharp. </br>
With the help of this project you won't have to transform .pt model to onnx, and can train your own model in C#.

## Feature

- Written in C# only.
- Train and predict your own model.
- Support Yolov5, Yolov8 and Yolov11 now.
- Support n/s/m/l/x size.
- Support LetterBox and Mosaic4 method for preprocessing images.
- Support Load PreTrained models from ultralytics/yolov5 (converted)


## Models

You can download yolov5 pre-trained models here.

| model | n| s | m | l | x |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yolov5 | [yolov5n](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov5n.bin) | [yolov5s](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov5s.bin) | [yolov5m](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov5m.bin) | [yolov5l](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov5l.bin) | [yolov5x](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov5x.bin) |
| yolov8 | [yolov5n](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov8n.bin) | [yolov5s](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov8s.bin) | [yolov5m](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov8m.bin) | [yolov5l](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov8l.bin) | [yolov5x](https://github.com/IntptrMax/YoloSharp/releases/download/v1.0.2/Yolov8x.bin) |

For example:

Use yolov5n pre-trained model to detect coco128

![image](https://github.com/user-attachments/assets/d32f7805-9f98-4530-bda6-43630c765159)


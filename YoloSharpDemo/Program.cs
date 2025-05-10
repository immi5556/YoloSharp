using ImageMagick;
using ImageMagick.Drawing;
using YoloSharp;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
            //string trainDataPath = @"..\..\..\Assets\DataSets\coco128-seg"; // Training data path, it should be the same as coco dataset.
            //string valDataPath = @"..\..\..\Assets\DataSets\coco128-seg"; // If valDataPath is "", it will use trainDataPath as validation data.
            //string outputPath = "result";    // Trained model output path.
            //string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-seg.bin"; // Pretrained model path.
            //string predictImagePath = @"..\..\..\Assets\TestImage\bus.jpg";
            //int batchSize = 16;
            //int sortCount = 80;
            //int epochs = 10;
            //float predictThreshold = 0.5f;x
            //float iouThreshold = 0.45f;


            string trainDataPath = @"C:\Immi\sample\ml.net\car-damage-dataset\archive_3\train"; // Training data path, it should be the same as coco dataset.
            string valDataPath = @"C:\Immi\sample\ml.net\car-damage-dataset\archive_3\valid"; // If valDataPath is "", it will use trainDataPath as validation data.
            string outputPath = "result_car_damage_v3";    // Trained model output path.
            string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov11n-seg.bin"; // Pretrained model path.
                                                                                              //string predictImagePath = @"C:\Immi\sample\ml.net\yolo\datasets\carparts-seg\valid\images\new_7_png_jpg.rf.6be4e774157462beafcd5bf74c1e7d46.jpg";
                                                                                            //string predictImagePath = @"C:\Immi\sample\ml.net\yolo\datasets\carparts-seg\valid\images\new_7_png_jpg.rf.6be4e774157462beafcd5bf74c1e7d46.jpg";
            string predictImagePath = @"C:\Immi\sample\ml.net\yolo\datasets\carparts-seg\test\images\car4_jpg.rf.8978131a7b03be689c244641e42e1307.jpg";
            int batchSize = 16;
            int sortCount = 80;
            int epochs = 100; 
            float predictThreshold = 0.5f;
            float iouThreshold = 0.45f;


            YoloType yoloType = YoloType.Yolov11;
			DeviceType deviceType = DeviceType.CPU;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;

			MagickImage predictImage = new MagickImage(predictImagePath);

			//// Create predictor
			//Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);

			//// Train model
			//predictor.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);
			//predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: true);

			//// ImagePredict image
			//predictor.LoadModel(Path.Combine(outputPath, "best.bin"));
			//List<Predictor.PredictResult> predictResult = predictor.ImagePredict(predictImage, predictThreshold, iouThreshold);
			//var resultImage = predictImage.Clone();

			// Create segmenter
			Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			// Train model
			segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
			segmenter.LoadModel(Path.Combine(outputPath, "best.bin"), skipNcNotEqualLayers: false);

			// ImagePredict image
			var (predictResult, resultImage) = segmenter.ImagePredict(predictImage, predictThreshold, iouThreshold);

			if (predictResult.Count > 0)
			{
				var drawables = new Drawables()
					.StrokeColor(MagickColors.Red)
					.StrokeWidth(1)
					.FillColor(MagickColors.Transparent)
					.Font("Consolas")
					.FontPointSize(16)
					.TextAlignment(TextAlignment.Left);

				foreach (var result in predictResult)
				{
					drawables.Rectangle(result.X, result.Y, result.X + result.W, result.Y + result.H);
					string label = string.Format("Sort:{0}, Score:{1:F1}%", result.ClassID, result.Score * 100);
					drawables.Text(result.X, result.Y - 12, label);
					Console.WriteLine(label);
				}
				resultImage.Draw(drawables);
				resultImage.Write("pred_car_damage_v1.jpg");
			}

			Console.WriteLine();
			Console.WriteLine("ImagePredict done");
		}

	}
}


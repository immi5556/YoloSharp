using ImageMagick;
using ImageMagick.Drawing;
using YoloSharp;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string trainDataPath = @"..\..\..\Assets\DataSets\coco128"; // Training data path, it should be the same as coco dataset.
			string valDataPath = string.Empty; // If valDataPath is "", it will use trainDataPath as validation data.
			string outputPath = "result";    // Trained model output path.
			string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n.bin"; // Pretrained model path.
			string predictImagePath = @"..\..\..\Assets\TestImage\zidane.jpg";
			int batchSize = 16;
			int sortCount = 80;
			int epochs = 100;
			float predictThreshold = 0.4f;
			float iouThreshold = 0.4f;

			YoloType yoloType = YoloType.Yolov8;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;

			MagickImage predictImage = new MagickImage(predictImagePath);

			// Create predictor
			Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			// Train model
			predictor.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);
			predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: true);

			// ImagePredict image
			predictor.LoadModel(Path.Combine(outputPath, "best.bin"));
			List<Predictor.PredictResult> predictResult = predictor.ImagePredict(predictImage, predictThreshold, iouThreshold);
			var resultImage = predictImage.Clone();

			////Create segmenter
			//Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			//segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			//// Train model
			//segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
			//segmenter.LoadModel(Path.Combine(outputPath, "best.bin"));

			//// ImagePredict image
			//var (predictResult, resultImage) = segmenter.ImagePredict(predictImage, predictThreshold, iouThreshold);

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
				resultImage.Write("pred.jpg");
			}

			Console.WriteLine();
			Console.WriteLine("ImagePredict done");
		}

	}
}


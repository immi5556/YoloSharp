using System.Drawing;
using YoloSharp;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string trainDataPath = @"..\..\..\Assets\DataSets\coco128-seg"; // Training data path, it should be the same as coco dataset.
			string valDataPath = @"..\..\..\Assets\DataSets\coco128-seg"; // If valDataPath is "", it will use trainDataPath as validation data.
			string outputPath = "result";    // Trained model output path.
			string preTraindModelPath = @"..\..\..\Assets\PreTrainedModels\yolov11n-seg.bin"; // Pretrained model path.
			string predictImagePath = @"..\..\..\Assets\TestImage\bus.jpg";
			int batchSize = 16;
			int sortCount = 80;
			int epochs = 100;
			float predictThreshold = 0.5f;
			float iouThreshold = 0.45f;

			YoloType yoloType = YoloType.Yolov11;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;

			//// Create predictor
			//Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);

			//// Train model
			//predictor.LoadModel(preTraindModelPath, skipNcNotEqualLayers: true);
			//predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: true);

			////ImagePredict image
			//Bitmap bitmap = new Bitmap(predictImagePath);
			//predictor.LoadModel(Path.Combine(outputPath, "best.bin"));
			//var predictResult = predictor.ImagePredict(bitmap, predictThreshold, iouThreshold);

			// Create segmenter
			Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			segmenter.LoadModel(preTraindModelPath, skipNcNotEqualLayers: true);

			segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
			segmenter.LoadModel("output/best.bin");

			Bitmap testBitmap = new Bitmap(predictImagePath);
			var (predictResult, bitmap) = segmenter.ImagePredict(testBitmap, predictThreshold, iouThreshold);

			if (predictResult.Count > 0)
			{
				// Draw results
				Graphics g = Graphics.FromImage(bitmap);
				foreach (var result in predictResult)
				{
					Point point = new Point(result.X, result.Y);
					string str = string.Format("Sort:{0}, Score:{1:F1}%", result.ClassID, result.Score * 100);
					g.DrawRectangle(Pens.Red, new Rectangle(point, new Size(result.W, result.H)));
					g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
					Console.WriteLine(str);
				}
				g.Save();
				bitmap.Save("pred.jpg");
			}

			Console.WriteLine();
			Console.WriteLine("ImagePredict done");
		}

	}
}


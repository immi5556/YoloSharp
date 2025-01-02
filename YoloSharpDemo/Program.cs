using System.Drawing;
using YoloSharp;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string trainDataPath = @"..\..\..\Assets\coco128"; // Training data path, it should be the same as coco dataset.
			string valDataPath = @"..\..\..\Assets\coco128"; // If valDataPath is "", it will use trainDataPath as validation data.
			string outputPath = "result";    // Trained model output path.
			string preTraindModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n.bin"; // Pretrained model path.
			string predictImagePath = @"..\..\..\Assets\TestImage\zidane.jpg";
			int batchSize = 8;
			int sortCount = 80;
			int epochs = 100;
			float predictThreshold = 0.3f;
			float iouThreshold = 0.5f;

			YoloType yoloType = YoloType.Yolov8;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;

			Bitmap inputBitmap = new Bitmap(predictImagePath);
			
			Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);

			// Train model
			predictor.Train(trainDataPath, valDataPath, preTraindModelPath: preTraindModelPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs);

			// Predict image
			var results = predictor.ImagePredict(inputBitmap, Path.Combine(outputPath, "best.bin"), predictThreshold, iouThreshold);
			
			Graphics g = Graphics.FromImage(inputBitmap);
			foreach (var result in results)
			{
				Point point = new Point(result.X - result.W / 2, result.Y - result.H / 2);
				string str = string.Format("Sort:{0}, Score:{1:F1}%", result.ClassID, result.Score * 100);
				g.DrawRectangle(Pens.Red, new Rectangle(point, new Size(result.W, result.H)));
				g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
				Console.WriteLine(str);
			}
			g.Save();
			inputBitmap.Save("pred.jpg");
			Console.WriteLine();
			Console.WriteLine("Predict done");
		}

	}
}


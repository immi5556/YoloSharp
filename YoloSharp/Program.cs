using System.Drawing;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace YoloSharp
{
	internal class Program
	{
		private static string dataPath = @"..\..\..\Assets\coco128";
		private static int sortCount = 80;
		private static int epochs = 100;
		private static float lr = 0.0001f;
		private static int imageSize = 640;
		private static Device device = CUDA;
		private static ScalarType scalarType = ScalarType.Float32;
		//private static Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount, Yolo.YoloSize.n);
		private static Yolo.Yolov5u yolo = new Yolo.Yolov5u(sortCount, Yolo.YoloSize.n);
		//private static Yolo.Yolov8 yolo = new Yolo.Yolov8(sortCount, Yolo.YoloSize.n);
		//private static Yolo.Yolov11 yolo = new Yolo.Yolov11(sortCount, Yolo.YoloSize.n);

		static void Main(string[] args)
		{
			Train();
			ImagePredict();
		}


		private static void Train()
		{
			YoloDataset yoloDataset = new YoloDataset(dataPath, imageSize, deviceType: device.type, useMosaic: true);
			DataLoader loader = new DataLoader(yoloDataset, 16, num_worker: 0, shuffle: true, device: device);
			//Loss.Yolov5DetectionLoss loss = new Loss.Yolov5DetectionLoss(sortCount).to(device);
			Loss.Yolov8DetectionLoss loss = new Loss.Yolov8DetectionLoss(sortCount).to(device);
			yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5u\Yolov5nu.bin");
			yolo.train();
			yolo.to(device, scalarType);

			var optimizer = optim.AdamW(yolo.parameters(), lr, weight_decay: 0.0005);

			float tempLoss = float.MaxValue;

			for (int epoch = 0; epoch < epochs; epoch++)
			{
				int step = 0;
				foreach (var data in loader)
				{
					step++;
					long[] indexs = data["index"].data<long>().ToArray();
					Tensor[] images = new Tensor[indexs.Length];
					Tensor[] labels = new Tensor[indexs.Length];
					for (int i = 0; i < indexs.Length; i++)
					{
						var (img, lb) = yoloDataset.GetDataTensor(indexs[i]);
						images[i] = img.to(scalarType, device);
						labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: scalarType, device: lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor imageTensor = concat(images);

					Tensor labelTensor = concat(labels);

					if (labelTensor.shape[0] == 0)
					{
						continue;
					}

					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list.ToArray(), labelTensor);
					ls.backward();
					optimizer.step();
					optimizer.zero_grad();
					Console.WriteLine($"Epoch {epoch}, Step {step}/{loader.Count} , Loss: {ls.ToSingle()}");

					if (tempLoss > ls.ToSingle())
					{
						if (!Directory.Exists("result"))
						{
							Directory.CreateDirectory("result");
						}
						yolo.save(Path.Combine("result", "best.bin"));
						tempLoss = ls.ToSingle();
					}
					yolo.save(Path.Combine("result", "last.bin"));
					GC.Collect();
				}
			}
			if (!Directory.Exists("result"))
			{
				Directory.CreateDirectory("result");
			}
			yolo.save(Path.Combine("result", "yolo_last.bin"));
		}

		private static void ImagePredict()
		{
			float PredictThreshold = 0.3f;
			float IouThreshold = 0.5f;

			string orgImagePath = @"..\..\..\Assets\TestImage\zidane.jpg";
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			Tensor orgImage = torchvision.io.read_image(orgImagePath).to(scalarType, device).unsqueeze(0) / 255.0f;
			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = torch.nn.functional.pad(orgImage, [0, padWidth, 0, padHeight], PaddingModes.Zeros);
			//yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov11\Yolov11n.bin");
			yolo.load(@"result/best.bin");

			yolo.to(device, scalarType);
			yolo.eval();
			//yolo.train(true);

			Tensor[] tensors = yolo.forward(input);

			// You should selet correct Predictor: Predict.Yolov5Predict or Predict.Yolov8Predict 
			//Predict.Yolov5Predict predict = new Predict.Yolov5Predict();	
			Predict.Yolov8Predict predict = new Predict.Yolov8Predict(PredictThreshold, IouThreshold);
			var results = predict.Predict(tensors[0]);

			Bitmap bitmap = new Bitmap(orgImagePath);
			Graphics g = Graphics.FromImage(bitmap);
			foreach (var result in results)
			{
				Point point = new Point(result.x - result.w / 2, result.y - result.h / 2);
				Rectangle rect = new Rectangle(point, new System.Drawing.Size(result.w, result.h));
				string str = string.Format("Sort:{0}, Score:{1:F1}%", result.sort, result.score * 100);
				Console.WriteLine(str);
				g.DrawRectangle(Pens.Red, rect);
				g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
			}
			g.Save();
			bitmap.Save("result.jpg");

			Console.WriteLine();
			Console.WriteLine("Predict done");


		}

	}

}


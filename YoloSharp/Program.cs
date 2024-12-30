using System.Drawing;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	// Yolov5u Yolov8 Yolov11 have the same predict and loss method, so we can use the same predict method for them.
	// But yolov5 has a different model structure, so we need to use different model. 

	internal class Program
	{
		private static string dataPath = @"..\..\..\Assets\coco128";
		private static int sortCount = 80;
		private static int epochs = 100;
		private static float lr = 0.0001f;
		private static int imageSize = 640;
		private static Device device = CUDA;
		private static ScalarType scalarType = ScalarType.Float16;
		private static Module<Tensor, Tensor[]> yolo = new Yolo.Yolov5u(sortCount, Yolo.YoloSize.n);

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
			Loss.YolovDetectionLoss loss = new Loss.YolovDetectionLoss(sortCount).to(device);
			yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5u\Yolov5nu.bin");
			yolo.train();
			yolo.to(device, scalarType);

			var optimizer = optim.SGD(yolo.parameters(), lr, weight_decay: 0.0005);

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
			float PredictThreshold = 0.5f;
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
			//yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5u\Yolov5nu.bin");
			yolo.load(@"result/best.bin");

			yolo.to(device, scalarType);
			yolo.eval();

			Tensor[] tensors = yolo.forward(input);

			// You should selet correct Predictor: Predict.Yolov5Predict or Predict.YoloPredict 
			//Predict.Yolov5Predict predict = new Predict.Yolov5Predict();	
			Predict.YoloPredict predict = new Predict.YoloPredict(PredictThreshold, IouThreshold);
			Tensor results = predict.Predict(tensors[0]);
			Bitmap bitmap = new Bitmap(orgImagePath);
			Graphics g = Graphics.FromImage(bitmap);
			if (results.shape[0]>0)
			{
				for (int i = 0; i < results.shape[0]; i++)
				{
					int x = results[i, 0].ToInt32();
					int y = results[i, 1].ToInt32();
					int rw = results[i, 2].ToInt32();
					int rh = results[i, 3].ToInt32();
					float score = results[i, 4].ToSingle();
					int sort = results[i, 5].ToInt32();
					Point point = new Point(x - rw / 2, y - rh / 2);
					Rectangle rect = new Rectangle(point, new System.Drawing.Size(rw, rh));
					string str = string.Format("Sort:{0}, Score:{1:F1}%", sort, score * 100);
					Console.WriteLine(str);
					g.DrawRectangle(Pens.Red, rect);
					g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
				}
			}
			g.Save();
			bitmap.Save("result.jpg");

			Console.WriteLine();
			Console.WriteLine("Predict done");


		}

	}

}


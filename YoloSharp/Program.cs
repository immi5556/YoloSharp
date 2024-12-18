using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
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
		private static Yolo.Yolov8 yolo = new Yolo.Yolov8(sortCount, Yolo.YoloSize.n);

		static void Main(string[] args)
		{
			//Train();
			ImagePredict();
		}

		private static void Train()
		{
			YoloDataset yoloDataset = new YoloDataset(dataPath, imageSize, deviceType: device.type, useMosaic: false);
			DataLoader loader = new DataLoader(yoloDataset, 16, num_worker: 32, shuffle: true, device: device);
			//Loss.Yolov5DetectionLoss loss = new Loss.Yolov5DetectionLoss(sortCount).to(device);
			Loss.Yolov8DetectionLoss loss = new Loss.Yolov8DetectionLoss(sortCount).to(device);
			yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov8\Yolov8n.bin");
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
			int predictIndex = 10;
			float PredictThreshold = 0.1f;
			float IouThreshold = 0.5f;

			YoloDataset yoloDataset = new YoloDataset(dataPath, useMosaic: false);
			Tensor input = yoloDataset.GetDataTensor(predictIndex).Item1.to(scalarType, device);
			//yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5\Yolov5n.bin");
			yolo.to(device, scalarType);


			yolo.load(@"result/best.bin");
			//yolo.load(@"yolov8n.bin");
			yolo.eval();

			Tensor[] tensors = yolo.forward(input);

			// You should selet correct Predictor: Predict.Yolov5Predict or Predict.Yolov8Predict 
			Predict.Yolov8Predict predict = new Predict.Yolov8Predict();
			var results = predict.Predict(tensors[0]);

			Tensor orgImg = (input.squeeze(0) * 255).@byte().cpu();
			orgImg = orgImg.index_select(0, tensor(new long[] { 2, 1, 0 }));
			orgImg = orgImg.permute([1, 2, 0]).contiguous();
			Bitmap bitmap = new Bitmap((int)orgImg.shape[1], (int)orgImg.shape[0]);
			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
			Marshal.Copy(orgImg.bytes.ToArray(), 0, bitmapData.Scan0, bitmapData.Stride * bitmapData.Height);
			bitmap.UnlockBits(bitmapData);

			Graphics g = Graphics.FromImage(bitmap);
			foreach (var result in results)
			{
				Point point = new Point(result.x - result.w / 2, result.y - result.h / 2);
				Rectangle rect = new Rectangle(point, new System.Drawing.Size(result.w, result.h));
				string str = string.Format("Sort:{0}, Score:{1:F1}%", result.sort, result.score * 100);
				g.DrawRectangle(Pens.Red, rect);
				g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
			}
			g.Save();
			bitmap.Save("bitmap.jpg");
		}

	}

}


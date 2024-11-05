using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Yolov5
{
	internal class Program
	{
		private static string dataPath = @"..\..\..\Assets\coco128";
		private static int sortCount = 80;
		private static int epochs = 3000;
		private static float lr = 0.01f;
		private static int imageSize = 640;
		private static Device device = torch.CUDA;

		static void Main(string[] args)
		{
			Train();
			Predict();
		}

		private static void Train()
		{
			YoloDataset yoloDataset = new YoloDataset(dataPath, imageSize, deviceType: device.type, useMosaic: true);
			DataLoader loader = new DataLoader(yoloDataset, 16, num_worker: 32, shuffle: true, device: device);
			Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount).to(device);
			Loss.Yolov5Loss loss = new Loss.Yolov5Loss(sortCount).to(device);

			var optimizer = optim.SGD(yolo.parameters(), learningRate: lr, weight_decay: 0.0005);

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
						images[i] = img.to(device);
						labels[i] = torch.full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, ScalarType.Float32,device:lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor imageTensor = torch.concat(images);
					Tensor labelTensor = torch.concat(labels);

					if (labelTensor.shape[0] == 0)
					{
						continue;
					}

					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list.ToArray(), labelTensor);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();

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
				}
			}
			if (!Directory.Exists("result"))
			{
				Directory.CreateDirectory("result");
			}
			yolo.save(Path.Combine("result", "yolo_last.bin"));
		}

		private static void Predict()
		{
			int predictIndex = 0;
			float PredictThreshold = 0.5f;
			float ObjectThreshold = 0.5f;
			float NmsThreshold = 0.4f;
			double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];

			List<Result> results = new List<Result>();

			YoloDataset yoloDataset = new YoloDataset(dataPath, useMosaic: false);
			Tensor input = yoloDataset.GetDataTensor(predictIndex).Item1;
			Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount).cuda();
			yolo.load("result/last.bin");
			yolo.eval();
			Tensor[] tensors = yolo.forward(input.cuda());

			Tensor resultTensor = tensors[0].squeeze(0);
			Tensor predScores = resultTensor[TensorIndex.Ellipsis, TensorIndex.Slice(4, 5)];
			var preds = predScores > PredictThreshold;

			var indices = torch.nonzero(preds)[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1)].squeeze(-1);
			var rResult = resultTensor[indices];
			if (rResult.shape[0] > 0)
			{
				for (int i = 0; i < rResult.shape[0]; i++)
				{
					var scores = rResult[i][TensorIndex.Ellipsis, TensorIndex.Slice(5, rResult.shape[1] + 1)];
					var (maxScore, index) = scores.max(0);
					if (maxScore.ToSingle() > ObjectThreshold)
					{
						results.Add(new Result
						{
							sort = (int)index.ToInt64(),
							score = maxScore.ToSingle(),
							x = rResult[i][0].ToSingle(),
							y = rResult[i][1].ToSingle(),
							w = rResult[i][2].ToSingle(),
							h = rResult[i][3].ToSingle(),
						});
					}
				}

				results = NMS(results, NmsThreshold);
			}

			Tensor mean = torch.tensor(means).view(3, 1, 1);
			Tensor std = torch.tensor(stdevs).view(3, 1, 1);
			Tensor orgImg = ((input.cpu().squeeze(0) * std + mean) * 255.0f).@byte();

			//Tensor orgImg = (input.squeeze(0) * 255.0f).@byte().cpu();
			orgImg = orgImg.index_select(0, tensor(new long[] { 2, 1, 0 }));
			orgImg = orgImg.permute([1, 2, 0]).contiguous();
			Bitmap bitmap = new Bitmap(imageSize, imageSize);
			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, imageSize, imageSize), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
			Marshal.Copy(orgImg.bytes.ToArray(), 0, bitmapData.Scan0, imageSize * imageSize * 3);
			bitmap.UnlockBits(bitmapData);


			Graphics g = Graphics.FromImage(bitmap);
			foreach (Result result in results)
			{
				Point point = new Point((int)(result.x - result.w / 2), (int)(result.y - result.h / 2));
				Rectangle rect = new Rectangle(point, new System.Drawing.Size((int)result.w, (int)result.h));
				string str = string.Format("Sort:{0}, Score:{1}", result.sort, result.score);
				g.DrawRectangle(Pens.Red, rect);
				g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);

			}
			g.Save();
			bitmap.Save("bitmap.jpg");

		}


		private static Bitmap DrawImageWithLabels(Tensor img, Tensor labels)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (img.shape.Length == 4)
			{
				img = img.squeeze(0);
			}
			int width = (int)img.shape[2];
			int height = (int)img.shape[1];
			img = (img * 255.0f).to(ScalarType.Byte);
			img = img.cpu();

			torchvision.io.write_jpeg(img, "img.jpg");

			byte[] tensorBytes = img.bytes.ToArray();
			Bitmap bitmap = new Bitmap(width, height);
			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

			byte[] bytes = new byte[width * height * 3];
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					bytes[h * width * 3 + w * 3 + 0] = tensorBytes[2 * (height * width) + width * h + w];
					bytes[h * width * 3 + w * 3 + 1] = tensorBytes[1 * (height * width) + width * h + w];
					bytes[h * width * 3 + w * 3 + 2] = tensorBytes[0 * (height * width) + width * h + w];

				}
			}

			Marshal.Copy(bytes, 0, bitmapData.Scan0, width * height * 3);
			bitmap.UnlockBits(bitmapData);

			Graphics g = Graphics.FromImage(bitmap);

			for (int i = 0; i < labels.shape[0]; i++)
			{
				int w = (int)(labels[i][3].ToSingle() * width);
				int h = (int)(labels[i][4].ToSingle() * height);
				int x = (int)(labels[i][1].ToSingle() * width - w / 2);
				int y = (int)(labels[i][2].ToSingle() * height - h / 2);
				int sort = labels[i][0].ToInt32();
				Point point = new Point(x, y);
				Rectangle rect = new Rectangle(x, y, w, h);
				g.DrawRectangle(Pens.Red, rect);
				string str = string.Format("Sort:{0}", sort);
				g.DrawString(str, new Font(FontFamily.GenericMonospace, 10), new SolidBrush(Color.Red), point);
			}
			g.Save();
			return bitmap;
		}



		private static List<Result> NMS(List<Result> orgResults, float threshold = 0.5f)
		{
			List<Result> results = new List<Result>();
			List<int> sorts = new List<int>();
			foreach (Result re in orgResults)
			{
				if (!sorts.Contains(re.sort))
				{
					sorts.Add(re.sort);
				}
			}
			foreach (int sort in sorts)
			{
				List<Result> list = orgResults.FindAll(re => re.sort == sort);
				float[] data = new float[list.Count * 4];
				float[] scores = new float[list.Count];
				for (int i = 0; i < list.Count; i++)
				{
					data[4 * i + 0] = list[i].x - list[i].w / 2;
					data[4 * i + 1] = list[i].y - list[i].h / 2;
					data[4 * i + 2] = list[i].x + list[i].w / 2;
					data[4 * i + 3] = list[i].y + list[i].h / 2;
					scores[i] = list[i].score;
				}

				Tensor boxes = torch.tensor(data);
				boxes = boxes.view([list.Count, 4]);
				Tensor scr = torch.tensor(scores);
				Tensor r = torchvision.ops.nms(boxes, scr, threshold);
				long[] nmsIndexs = r.data<long>().ToArray();
				foreach (long nms in nmsIndexs)
				{
					results.Add(list[(int)nms]);
				}
			}
			return results;
		}

		class Result
		{
			public float score;
			public int sort;
			public float x;
			public float y;
			public float w;
			public float h;
		}

	}
}

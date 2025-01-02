using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Predictor
	{
		public class PredictResult
		{
			public int ClassID;
			public float Score;
			public int X;
			public int Y;
			public int W;
			public int H;
		}



		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Tensor, (Tensor, Tensor)> loss;
		private Module predict;
		private torch.Device device;
		private torch.ScalarType dtype;
		private int socrCount;
		private YoloType yoloType;

		public Predictor(int socrCount = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
		{
			this.device = new torch.Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.socrCount = socrCount;
			this.yoloType = yoloType;

			yolo = yoloType switch
			{
				YoloType.Yolov5 => new Yolov5(socrCount, yoloSize),
				YoloType.Yolov5u => new Yolov5u(socrCount, yoloSize),
				YoloType.Yolov8 => new Yolov8(socrCount, yoloSize),
				YoloType.Yolov11 => new Yolov11(socrCount, yoloSize),
				_ => throw new NotImplementedException(),
			};

			yolo.to(this.device, this.dtype);

			loss = yoloType switch
			{
				YoloType.Yolov5 => new Loss.Yolov5DetectionLoss(this.socrCount),
				YoloType.Yolov5u => new Loss.YolovDetectionLoss(this.socrCount),
				YoloType.Yolov8 => new Loss.YolovDetectionLoss(this.socrCount),
				YoloType.Yolov11 => new Loss.YolovDetectionLoss(this.socrCount),
				_ => throw new NotImplementedException(),
			};
			loss = loss.to(this.device, this.dtype);

			predict = yoloType switch
			{
				YoloType.Yolov5 => new Predict.Yolov5Predict(),
				YoloType.Yolov5u => new Predict.YoloPredict(),
				YoloType.Yolov8 => new Predict.YoloPredict(),
				YoloType.Yolov11 => new Predict.YoloPredict(),
				_ => throw new NotImplementedException(),
			};
		}

		public void Train(string trainDataPath, string valDataPath = "", string outputPath = "output", string preTraindModelPath = "", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, bool useMosaic = true)
		{
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");
			if (!string.IsNullOrEmpty(preTraindModelPath))
			{
				ModelLoad(preTraindModelPath);
			}

			YoloDataset trainDataSet = new YoloDataset(trainDataPath, imageSize, deviceType: this.device.type, useMosaic: useMosaic);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}
			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;
			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr);
			float tempLoss = float.MaxValue;
			Console.WriteLine("Start Training...");
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				int step = 0;
				foreach (var data in trainDataLoader)
				{
					step++;
					long[] indexs = data["index"].data<long>().ToArray();
					Tensor[] images = new Tensor[indexs.Length];
					Tensor[] labels = new Tensor[indexs.Length];
					for (int i = 0; i < indexs.Length; i++)
					{
						var (img, lb) = trainDataSet.GetDataTensor(indexs[i]);
						images[i] = img.to(dtype, device);
						labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor imageTensor = concat(images);
					Tensor labelTensor = concat(labels);
					if (labelTensor.shape[0] == 0)
					{
						continue;
					}
					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list, labelTensor);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();
					Console.WriteLine($"Process: Epoch {epoch}, Step/Total Step  {step}/{trainDataLoader.Count}");
				}
				Console.Write("Do val now... ");
				float valLoss = Val(valDataPath, imageSize);
				Console.WriteLine($"Epoch {epoch}, Val Loss: {valLoss}");
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
				yolo.save(Path.Combine(outputPath, "last.bin"));
				if (tempLoss > valLoss)
				{
					yolo.save(Path.Combine(outputPath, "best.bin"));
					tempLoss = valLoss;
				}
			}
			Console.WriteLine("Train Done.");
		}

		private float Val(string valDataPath, int imageSize = 640)
		{
			YoloDataset yoloDataset = new YoloDataset(valDataPath, imageSize, deviceType: this.device.type, useMosaic: false);
			DataLoader loader = new DataLoader(yoloDataset, 4, num_worker: 0, shuffle: true, device: device);

			float lossValue = float.MaxValue;
			foreach (var data in loader)
			{
				long[] indexs = data["index"].data<long>().ToArray();
				Tensor[] images = new Tensor[indexs.Length];
				Tensor[] labels = new Tensor[indexs.Length];
				for (int i = 0; i < indexs.Length; i++)
				{
					var (img, lb) = yoloDataset.GetDataTensor(indexs[i]);
					images[i] = img.to(this.dtype, device);
					labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
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
				if (lossValue == float.MaxValue)
				{
					lossValue = ls.ToSingle();
				}
				else
				{
					lossValue = lossValue + ls.ToSingle();
				}
			}
			lossValue = lossValue / yoloDataset.Count;
			return lossValue;
		}

		public List<PredictResult> ImagePredict(Bitmap image, string modelPath, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			byte[] buffer = new byte[image.Width * image.Height * 3];
			BitmapData data = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
			Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);
			image.UnlockBits(data);
			Tensor orgImage = torch.tensor(buffer.ToArray()).view([image.Height, image.Width, 3]).permute([2, 0, 1]).contiguous().to(dtype, device);
			orgImage = torch.stack(new Tensor[] { orgImage[2], orgImage[1], orgImage[0] }, dim: 0).unsqueeze(0) / 255.0f;
			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = torch.nn.functional.pad(orgImage, [0, padWidth, 0, padHeight], PaddingModes.Zeros);
			ModelLoad(modelPath);
			yolo.eval();

			Tensor[] tensors = yolo.forward(input);

			Predict.YoloPredict predict = new Predict.YoloPredict(PredictThreshold, IouThreshold);
			Tensor results = predict.forward(tensors[0]);
			List<PredictResult> predResults = new List<PredictResult>();
			for (int i = 0; i < results.shape[0]; i++)
			{
				int x = results[i, 0].ToInt32();
				int y = results[i, 1].ToInt32();
				int rw = results[i, 2].ToInt32();
				int rh = results[i, 3].ToInt32();
				float score = results[i, 4].ToSingle();
				int sort = results[i, 5].ToInt32();

				predResults.Add(new PredictResult()
				{
					ClassID = sort,
					Score = score,
					X = x,
					Y = y,
					W = rw,
					H = rh
				});
			}
			return predResults;
		}

		private void ModelLoad(string path)
		{
			using FileStream input = File.OpenRead(path);
			using BinaryReader reader = new BinaryReader(input);
			int tensorCount = (int)Decode(reader);
			if (tensorCount != yolo.state_dict().Count)
			{
				Console.WriteLine("Mismatched tensor count while loading. Make sure that the model you are loading into is exactly the same as the origin.");
				Console.WriteLine("Model will run with random weight.");
			}
			else
			{
				string str = reader.ReadString();
				int typeIndex = (int)(sbyte)Decode(reader);
				if (typeIndex != (int)dtype)
				{
					yolo.to((torch.ScalarType)typeIndex);
					yolo.load(path);
					yolo.to(dtype);
				}
				else
				{
					yolo.load(path);
				}

			}
		}


		private long Decode(BinaryReader reader)
		{
			long num = 0L;
			int num2 = 0;
			while (true)
			{
				long num3 = reader.ReadByte();
				num += (num3 & 0x7F) << num2 * 7;
				if ((num3 & 0x80) == 0L)
				{
					break;
				}

				num2++;
			}

			return num;
		}

	}
}

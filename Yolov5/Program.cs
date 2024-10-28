using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Yolov5
{
	internal class Program
	{
		private static string dataPath = @"..\..\..\Assets\coco128";
		private static int sortCount = 80;
		private static int epochs = 1000000;
		private static float lr = 0.001f;

		static void Main(string[] args)
		{
			Train();
			//Predict();
		}

		private static void Train()
		{
			YoloDataset yoloDataset = new YoloDataset(dataPath);
			DataLoader loader = new DataLoader(yoloDataset, 32, num_worker: 32, shuffle: true, device: CUDA);
			Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount).cuda();
			Loss.Yolov5Loss loss = new Loss.Yolov5Loss(sortCount).cuda();

			var optimizer = optim.Adam(yolo.parameters(), lr: lr, weight_decay: 0.0005);

			float tempLoss = float.MaxValue;

			for (int epoch = 0; epoch < epochs; epoch++)
			{
				int step = 0;
				foreach (var data in loader)
				{
					step++;
					Tensor imageTensor = data["image"].cuda();
					Tensor indexTensor = data["index"];
					long[] indexs = indexTensor.data<long>().ToArray();
					Tensor[] labels = new Tensor[indexs.Length];
					for (int i = 0; i < labels.Length; i++)
					{
						Tensor lb = yoloDataset.GetLabelTensor(indexs[i]);
						labels[i] = torch.full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, ScalarType.Float32);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor labelTensor = torch.concat(labels);
					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list.ToArray(), labelTensor.cuda());
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
			int predictIndex = 1;
			float PredictThreshold = 0.5f;
			float ObjectThreshold = 0.9f;
			float NmsThreshold = 0.4f;
			List<Result> results = new List<Result>();

			YoloDataset yoloDataset = new YoloDataset(dataPath);
			Tensor input = yoloDataset.GetTensor(predictIndex)["image"].unsqueeze(0);

			Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount).cuda();
			yolo.load("result/best.bin");
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

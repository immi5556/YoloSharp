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
		private static int epochs = 300000;
		private static float lr = 0.01f;
		private static int imageSize = 640;
		private static Device device = CUDA;
		private static ScalarType scalarType = ScalarType.Float32;
		private static Yolo.Yolov5 yolo = new Yolo.Yolov5(sortCount, Yolo.YoloSize.n);

		static void Main(string[] args)
		{
			//Train();
			Predict();
		}

		private static void Train()
		{
			YoloDataset yoloDataset = new YoloDataset(dataPath, imageSize, deviceType: device.type, useMosaic: true);
			DataLoader loader = new DataLoader(yoloDataset, 16, num_worker: 32, shuffle: true, device: device);
			Loss.Yolov5Loss loss = new Loss.Yolov5Loss(sortCount).to(device);
			yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5\Yolov5n.bin");
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

		private static void Predict()
		{
			int predictIndex = 15;
			float PredictThreshold = 0.25f;
			float IouThreshold = 0.5f;

			List<Result> results = new List<Result>();

			YoloDataset yoloDataset = new YoloDataset(dataPath, useMosaic: false);
			Tensor input = yoloDataset.GetDataTensor(predictIndex).Item1.to(scalarType, device);
			yolo.to(ScalarType.Float32).load(@"..\..\..\Assets\models\Yolov5\Yolov5n.bin");
			yolo.to(device, scalarType);
			//yolo.load(@"result/best.bin");
			yolo.eval();

			Tensor[] tensors = yolo.forward(input);
			var re = NonMaxSuppression(tensors[0], PredictThreshold, IouThreshold);

			if (re[0].numel() > 0)
			{
				for (int i = 0; i < re[0].shape[0]; i++)
				{
					results.Add(new Result
					{
						x = re[0][i][0].ToInt32(),
						y = re[0][i][1].ToInt32(),
						w = re[0][i][2].ToInt32(),
						h = re[0][i][3].ToInt32(),
						score = re[0][i][4].ToSingle(),
						sort = re[0][i][5].ToInt32(),
					});
				}
			}

			Tensor orgImg = (input.squeeze(0) * 255).@byte().cpu();
			orgImg = orgImg.index_select(0, tensor(new long[] { 2, 1, 0 }));
			orgImg = orgImg.permute([1, 2, 0]).contiguous();
			Bitmap bitmap = new Bitmap((int)orgImg.shape[1], (int)orgImg.shape[0]);
			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
			Marshal.Copy(orgImg.bytes.ToArray(), 0, bitmapData.Scan0, bitmapData.Stride * bitmapData.Height);
			bitmap.UnlockBits(bitmapData);

			Graphics g = Graphics.FromImage(bitmap);
			foreach (Result result in results)
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


		class Result
		{
			public float score;
			public int sort;
			public int x;
			public int y;
			public int w;
			public int h;
		}


		public static List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
		{
			// Checks
			if (confThreshold < 0 || confThreshold > 1)
			{
				throw new ArgumentException($"Invalid Confidence threshold {confThreshold}, valid values are between 0.0 and 1.0");
			}
			if (iouThreshold < 0 || iouThreshold > 1)
			{
				throw new ArgumentException($"Invalid IoU {iouThreshold}, valid values are between 0.0 and 1.0");
			}

			var device = prediction.device;
			var scalType = prediction.dtype;

			var bs = prediction.shape[0]; // batch size
			var nc = prediction.shape[2] - nm - 5; // number of classes
			var xc = prediction[TensorIndex.Ellipsis, 4] > confThreshold; // candidates

			// Settings
			var max_wh = 7680; // maximum box width and height
			var max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
			var time_limit = 0.5f + 0.05f * bs; // seconds to quit after

			var t = DateTime.Now;
			var mi = 5 + nc; // mask start index
			var output = new List<Tensor>(new Tensor[bs]);
			for (int xi = 0; xi < bs; xi++)
			{
				var x = prediction[xi];
				x = x[xc[xi]]; // confidence

				// Compute conf
				x[TensorIndex.Ellipsis, TensorIndex.Slice(5, mi)] *= x[TensorIndex.Ellipsis, 4].unsqueeze(-1); // conf = obj_conf * cls_conf

				// Box/Mask
				var box = XYWH2XYXY(x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)]); // center_x, center_y, width, height) to (x1, y1, x2, y2)

				// Detections matrix nx6 (xyxy, conf, cls)

				var conf = x[TensorIndex.Colon, TensorIndex.Slice(5, mi)].max(1, true);
				var j = conf.indexes;
				x = torch.cat([box, conf.values, j.to_type(scalType)], 1)[conf.values.view(-1) > confThreshold];

				var n = x.shape[0]; // number of boxes
				if (n == 0)
				{
					continue;
				}

				x = x[x[TensorIndex.Ellipsis, 4].argsort(descending: true)][TensorIndex.Slice(0, max_nms)]; // sort by confidence and remove excess boxes

				// Batched NMS
				var c = x[TensorIndex.Ellipsis, 5].unsqueeze(-1) * (agnostic ? 0 : max_wh); // classes
				var boxes = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)] + c;
				var scores = x[TensorIndex.Ellipsis, 4];
				var i = torchvision.ops.nms(boxes, scores, iouThreshold); // NMS
				i = i[TensorIndex.Slice(0, max_det)]; // limit detections

				output[xi] = x[i];
				output[xi][TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)] = XYXY2XYWH(output[xi][TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)]);

				if ((DateTime.Now - t).TotalSeconds > time_limit)
				{
					Console.WriteLine($"WARNING ⚠️ NMS time limit {time_limit:F3}s exceeded");
					break; // time limit exceeded
				}
			}

			return output;
		}

		private static Tensor XYWH2XYXY(Tensor x)
		{
			Tensor y = x.clone();
			y[TensorIndex.Ellipsis, 0] = x[TensorIndex.Ellipsis, 0] - x[TensorIndex.Ellipsis, 2] / 2;  // top left x
			y[TensorIndex.Ellipsis, 1] = x[TensorIndex.Ellipsis, 1] - x[TensorIndex.Ellipsis, 3] / 2;  // top left y
			y[TensorIndex.Ellipsis, 2] = x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2] / 2; // bottom right x
			y[TensorIndex.Ellipsis, 3] = x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3] / 2; // bottom right y
			return y;
		}

		private static Tensor XYXY2XYWH(Tensor x)
		{
			var y = x.clone();
			y[TensorIndex.Ellipsis, 0] = (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2]) / 2;  // x center
			y[TensorIndex.Ellipsis, 1] = (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3]) / 2;// y center
			y[TensorIndex.Ellipsis, 2] = (x[TensorIndex.Ellipsis, 2] - x[TensorIndex.Ellipsis, 0]);  // width
			y[TensorIndex.Ellipsis, 3] = (x[TensorIndex.Ellipsis, 3] - x[TensorIndex.Ellipsis, 1]);  // height
			return y;
		}

	}

}


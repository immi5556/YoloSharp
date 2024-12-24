using TorchSharp;
using static TorchSharp.torch;

namespace YoloSharp
{
	public class Predict
	{
		public class Result
		{
			public float score;
			public int sort;
			public int x;
			public int y;
			public int w;
			public int h;
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

		public class Yolov5Predict
		{
			private readonly float PredictThreshold = 0.25f;
			private readonly float IouThreshold = 0.5f;
			public Yolov5Predict(float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				this.PredictThreshold = PredictThreshold;
				this.IouThreshold = IouThreshold;
			}

			public List<Result> Predict(Tensor tensor)
			{
				List<Result> results = new List<Result>();

				var re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (!Equals(re[0], null))
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

				return results;

			}

			private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
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
		}

		public class Yolov8Predict : Modules
		{
			private readonly float PredictThreshold = 0.25f;
			private readonly float IouThreshold = 0.5f;

			public Yolov8Predict(float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				this.PredictThreshold = PredictThreshold;
				this.IouThreshold = IouThreshold;
			}

			public List<Result> Predict(Tensor tensor)
			{
				List<Result> results = new List<Result>();
				var re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (!Equals(re[0], null))
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
				return results;

			}

			private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
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
				var nc = prediction.shape[1] - nm - 4; // number of classes
				var mi = 4 + nc; // mask start index
				var xc = prediction[TensorIndex.Colon, 4..(int)mi].amax(1) > confThreshold; // candidates

				prediction = prediction.transpose(1, 2);
				// Settings
				var max_wh = 7680; // maximum box width and height
				var max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
				var time_limit = 0.5f + 0.05f * bs; // seconds to quit after

				var t = DateTime.Now;

				var output = new List<Tensor>(new Tensor[bs]);
				for (int xi = 0; xi < bs; xi++)
				{
					var x = prediction[xi];
					x = x[xc[xi]]; // confidence

					// Box/Mask
					var box = XYWH2XYXY(x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)]); // center_x, center_y, width, height) to (x1, y1, x2, y2)

					// Detections matrix nx6 (xyxy, conf, cls)

					var conf = x[TensorIndex.Colon, TensorIndex.Slice(4, mi)].max(1, true);
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
		}

	}
}

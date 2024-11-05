using TorchSharp;
using static TorchSharp.torch;

namespace Yolov5
{
	internal class YoloDataset : torch.utils.data.Dataset
	{
		private static double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];
		private string rootPath = string.Empty;
		private int imageSize = 0;
		private List<string> imageFiles = new List<string>();
		private bool useMosaic = true;
		private Device device = torch.CUDA;

		public YoloDataset(string rootPath, int imageSize = 640, bool useMosaic = true, DeviceType deviceType = DeviceType.CUDA)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			this.rootPath = rootPath;
			string imagesFolder = Path.Combine(rootPath, "images");

			string[] imagesFileNames = Directory.GetFiles(imagesFolder, "*.*", SearchOption.AllDirectories).Where(file =>
			{
				string extension = Path.GetExtension(file).ToLower();
				return (extension == ".jpg" || extension == ".png" || extension == ".bmp");
			}).ToArray();
			foreach (string imageFileName in imagesFileNames)
			{
				string labelFileName = GetLabelFileNameFromImageName(imageFileName);
				if (!string.IsNullOrEmpty(labelFileName))
				{
					imageFiles.Add(imageFileName);
				}
			}
			this.imageSize = imageSize;
			this.useMosaic = useMosaic;
			this.device = new Device(deviceType);
		}

		private string GetLabelFileNameFromImageName(string imageFileName)
		{
			string imagesFolder = Path.Combine(rootPath, "images");
			string labelsFolder = Path.Combine(rootPath, "labels");
			string labelFileName = Path.ChangeExtension(imageFileName, ".txt").Replace(imagesFolder, labelsFolder);
			if (File.Exists(labelFileName))
			{
				return labelFileName;
			}
			else
			{
				return string.Empty;
			}
		}

		public override long Count => imageFiles.Count;

		public string GetFileNameByIndex(long index)
		{
			return imageFiles[(int)index];
		}

		public override Dictionary<string, torch.Tensor> GetTensor(long index)
		{
			Dictionary<string, torch.Tensor> outputs = new Dictionary<string, torch.Tensor>();
			outputs.Add("index", torch.tensor(index));
			return outputs;
		}

		public (torch.Tensor, torch.Tensor) GetTensorByLetterBox(long index)
		{
			string file = imageFiles[(int)index];
			Tensor orgImageTensor = torchvision.io.read_image(file).to(device);
			var (imgTensor, _, _) = Letterbox(orgImageTensor, imageSize, imageSize);
			Tensor lb = GetLetterBoxLabelTensor(index);
			return (imgTensor.unsqueeze(0), lb.to(imgTensor.device));
		}

		public (torch.Tensor, torch.Tensor) GetTensorByMosaic(long index)
		{
			var (img, lb) = load_mosaic(index);
			return (img.unsqueeze(0), lb.to(img.device));
		}

		public (torch.Tensor, torch.Tensor) GetDataTensor(long index)
		{
			var (image, label) = useMosaic ? GetTensorByMosaic(index) : GetTensorByLetterBox(index);
			if (image.shape.Length == 3)
			{
				image = image.unsqueeze(0);
			}
			image = torchvision.transforms.functional.normalize(image / 255.0f, means, stdevs);
			//image = image / 255.0f;
			return (image, label);
		}

		private (Tensor, float, int) Letterbox(Tensor image, int targetWidth, int targetHeight)
		{
			// 获取图像的原始尺寸
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			// 计算缩放比例
			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			// 计算缩放后的尺寸
			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			// 计算填充后的尺寸
			int padLeft = (targetWidth - scaledWidth) / 2;
			//int padRight = targetWidth - scaledWidth - padLeft;
			int padTop = (targetHeight - scaledHeight) / 2;
			//int padBottom = targetHeight - scaledHeight - padTop;

			// 缩放图像
			Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

			// 创建一个全零的张量，用于填充
			Tensor paddedImage = full([3, targetHeight, targetWidth], 114, image.dtype, image.device);

			// 将缩放后的图像放置在填充后的图像中心
			paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

			GC.Collect();

			return (paddedImage, scale, Math.Max(padLeft, padTop));
		}

		public Tensor GetLetterBoxLabelTensor(long index)
		{
			Tensor orgImageTensor = torchvision.io.read_image(imageFiles[(int)index]);
			var (imgTensor, scale, pad) = Letterbox(orgImageTensor, imageSize, imageSize);
			bool isWidthLonger = orgImageTensor.shape[2] > orgImageTensor.shape[1];


			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);

			float[,] labelArray = new float[lines.Length, 5];

			for (int i = 0; i < lines.Length; i++)
			{
				string[] labels = lines[i].Split(' ');
				labelArray[i, 0] = float.Parse(labels[0]);
				if (isWidthLonger)
				{
					labelArray[i, 1] = float.Parse(labels[1]);
					labelArray[i, 3] = float.Parse(labels[3]);

					labelArray[i, 2] = (float.Parse(labels[2]) * (imageSize - 2 * pad) + pad) / (float)imageSize;
					labelArray[i, 4] = float.Parse(labels[4]) * (imageSize - 2 * pad) / (float)imageSize;
				}
				else
				{
					labelArray[i, 1] = (float.Parse(labels[1]) * (imageSize - 2 * pad) + pad) / (float)imageSize;
					labelArray[i, 3] = float.Parse(labels[3]) * (imageSize - 2 * pad) / (float)imageSize;

					labelArray[i, 2] = float.Parse(labels[2]);
					labelArray[i, 4] = float.Parse(labels[4]);
				}

			}
			Tensor labelTensor = torch.tensor(labelArray);
			return labelTensor;

		}

		public Tensor GetOrgImage(long index)
		{
			return torchvision.io.read_image(imageFiles[(int)index]);
		}

		public (Tensor, Tensor) load_mosaic(long index)
		{
			int[] mosaic_border = [-320, -320];
			Int64[] indexs = Sample(index, 0, (int)Count, 4);
			Random random = new Random();
			int xc = random.Next(-mosaic_border[0], 2 * imageSize + mosaic_border[0]);
			int yc = random.Next(-mosaic_border[1], 2 * imageSize + mosaic_border[1]);

			var img4 = torch.full([3, imageSize * 2, imageSize * 2], 114, ScalarType.Byte, device); // base image with 4 tiles
			List<Tensor> label4 = new List<Tensor>();
			for (int i = 0; i < 4; i++)
			{
				int x1a = 0, y1a = 0, x2a = 0, y2a = 0, x1b = 0, y1b = 0, x2b = 0, y2b = 0;
				Tensor img = GetOrgImage(indexs[i]).to(device);
				//img = ResizeImage(img, resizeHeight);
				int h = (int)img.shape[1];
				int w = (int)img.shape[2];
				if (i == 0)  // top left
				{
					(x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), Math.Max(yc - h, 0), xc, yc);  // xmin, ymin, xmax, ymax (large image))
					(x1b, y1b, x2b, y2b) = (w - (x2a - x1a), h - (y2a - y1a), w, h); // xmin, ymin, xmax, ymax (small image);
				}
				else if (i == 1)  // top right
				{
					(x1a, y1a, x2a, y2a) = (xc, Math.Max(yc - h, 0), Math.Min(xc + w, imageSize * 2), yc);
					(x1b, y1b, x2b, y2b) = (0, h - (y2a - y1a), Math.Min(w, x2a - x1a), h);
				}
				else if (i == 2)  // bottom left
				{
					(x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), yc, xc, Math.Min(imageSize * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (w - (x2a - x1a), 0, w, Math.Min(y2a - y1a, h));
				}
				else if (i == 3) // bottom right
				{
					(x1a, y1a, x2a, y2a) = (xc, yc, Math.Min(xc + w, imageSize * 2), Math.Min(imageSize * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (0, 0, Math.Min(w, x2a - x1a), Math.Min(y2a - y1a, h));
				}
				img4[0..3, y1a..y2a, x1a..x2a] = img[0..3, y1b..y2b, x1b..x2b];

				int padw = x1a - x1b;
				int padh = y1a - y1b;

				Tensor labels = GetOrgLabelTensor(indexs[i]).to(device);
				labels[TensorIndex.Ellipsis, 1..5] = xywhn2xyxy(labels[TensorIndex.Ellipsis, 1..5], w, h, padw, padh);
				label4.Add(labels);
			}
			var labels4 = torch.concat(label4, 0);

			labels4[TensorIndex.Ellipsis, 1..5] = labels4[TensorIndex.Ellipsis, 1..5].clip(0, 2 * imageSize);

			var (im, targets) = random_perspective(img4, labels4, degrees: 0, translate: 0.1f, scale: 0.5f, shear: 0, perspective: 0.0f, mosaic_border[0], mosaic_border[1]);

			targets[TensorIndex.Ellipsis, 1..5] = xyxy2xywhn(targets[TensorIndex.Ellipsis, 1..5], w: (int)im.shape[1], h: (int)im.shape[2], clip: true, eps: 1e-3f);
			return (im, targets);
		}

		private Tensor ResizeImage(Tensor image, int targetWidth, int targetHeight)
		{
			// 获取图像的原始尺寸
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			// 计算缩放比例
			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			// 计算缩放后的尺寸
			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			return torchvision.transforms.functional.resize(image, scaledWidth, scaledHeight);
		}

		private Tensor ResizeImage(Tensor image, int targetSize)
		{
			return ResizeImage(image, targetSize, targetSize);
		}

		private Int64[] Sample(long orgIndex, int min, int max, int count)
		{
			Random random = new Random();
			List<Int64> list = new List<long>();
			while (list.Count < count - 1)
			{
				int number = random.Next(min, max);
				if (!list.Contains(number) && number != orgIndex)
				{
					if (random.NextSingle() > 0.5f)
					{
						list.Add(number);
					}
					else
					{
						list.Insert(0, number);
					}
				}
			}
			int i = random.Next(0, count);
			list.Insert(i, orgIndex);

			return list.ToArray();
		}

		private Tensor xywhn2xyxy(Tensor x, int w = 640, int h = 640, int padw = 0, int padh = 0)
		{
			//"""Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
			Tensor y = x.clone();
			y[TensorIndex.Ellipsis, 0] = w * (x[TensorIndex.Ellipsis, 0] - x[TensorIndex.Ellipsis, 2] / 2) + padw;  // top left x
			y[TensorIndex.Ellipsis, 1] = h * (x[TensorIndex.Ellipsis, 1] - x[TensorIndex.Ellipsis, 3] / 2) + padh;  // top left y
			y[TensorIndex.Ellipsis, 2] = w * (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2] / 2) + padw;  // bottom right x
			y[TensorIndex.Ellipsis, 3] = h * (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3] / 2) + padh;  // bottom right y
			return y;
		}

		private Tensor xyxy2xywhn(Tensor x, int w = 640, int h = 640, bool clip = false, float eps = 0.0f)
		{
			// Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right.
			if (clip)
			{
				x = clip_boxes(x, [h - eps, w - eps]);
			}
			var y = x.clone();
			y[TensorIndex.Ellipsis, 0] = ((x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2]) / 2) / w;  // x center
			y[TensorIndex.Ellipsis, 1] = ((x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3]) / 2) / h;// y center
			y[TensorIndex.Ellipsis, 2] = (x[TensorIndex.Ellipsis, 2] - x[TensorIndex.Ellipsis, 0]) / w;  // width
			y[TensorIndex.Ellipsis, 3] = (x[TensorIndex.Ellipsis, 3] - x[TensorIndex.Ellipsis, 1]) / h;  // height
			return y;
		}

		private Tensor clip_boxes(Tensor boxes, float[] shape)
		{
			// """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""

			boxes[TensorIndex.Ellipsis, 0] = boxes[TensorIndex.Ellipsis, 0].clamp_(0, shape[1]);  // x1
			boxes[TensorIndex.Ellipsis, 1] = boxes[TensorIndex.Ellipsis, 1].clamp_(0, shape[0]);  // y1
			boxes[TensorIndex.Ellipsis, 2] = boxes[TensorIndex.Ellipsis, 2].clamp_(0, shape[1]);  // x2
			boxes[TensorIndex.Ellipsis, 3] = boxes[TensorIndex.Ellipsis, 3].clamp_(0, shape[0]);  // y2
			return boxes;
		}

		private Tensor GetOrgLabelTensor(long index)
		{
			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);

			float[,] labelArray = new float[lines.Length, 5];

			for (int i = 0; i < lines.Length; i++)
			{
				string[] labels = lines[i].Split(' ');
				for (int j = 0; j < labels.Length; j++)
				{
					labelArray[i, j] = float.Parse(labels[j]);
				}
			}
			Tensor labelTensor = torch.tensor(labelArray);
			return labelTensor;
		}

		private (Tensor, Tensor) random_perspective(Tensor im, Tensor targets, int degrees = 10, float translate = 0.1f, float scale = 0.1f, int shear = 10, float perspective = 0.0f, int borderX = 0, int borderY = 0)
		{
			Device device = im.device;
			int height = (int)im.shape[1] + borderY * 2;
			int width = (int)im.shape[2] + borderX * 2;

			// Center
			Tensor C = torch.eye(3).to(device);
			C[0, 2] = -im.shape[2] / 2; // x translation (pixels)
			C[1, 2] = -im.shape[1] / 2; // y translation (pixels)

			//Perspective
			Tensor P = torch.eye(3).to(device);
			P[2, 0] = torch.rand(1).ToSingle() * 2 * perspective - perspective;   // x perspective (about y)
			P[2, 1] = torch.rand(1).ToSingle() * 2 * perspective - perspective;   // y perspective (about x)

			// Rotation and Scale
			float a = torch.rand(1).ToSingle() * 2 * degrees - degrees;
			float s = 1 + scale - torch.rand(1).ToSingle() * 2 * scale;

			Tensor R = GetRotationMatrix2D(angle: a, scale: s).to(device);

			// Shear
			Tensor S = torch.eye(3).to(device);
			S[0, 1] = Math.Tan((torch.rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // x shear (deg)
			S[1, 0] = Math.Tan((torch.rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // y shear (deg)

			// Translation
			Tensor T = torch.eye(3).to(device);
			T[0, 2] = (0.5f + translate - torch.rand(1).ToSingle() * 2 * translate) * width;    // x translation(pixels)
			T[1, 2] = (0.5f + translate - torch.rand(1).ToSingle() * 2 * translate) * height;   // y translation(pixels)

			var M = T.mm(S).mm(R).mm(P).mm(C);

			Tensor outTensor = torch.zeros([imageSize, imageSize, 3], ScalarType.Byte).to(device);
			if (borderX != 0 || borderY != 0 || M.bytes != torch.eye(3).bytes)
			{
				if (perspective != 0)
				{
					//im = WarpPerspective(im, M, width, height, (114, 114, 114));
				}
				else
				{
					// 提取仿射变换的参数
					var shearParams = new List<float> { S[0, 1].ToSingle(), S[1, 0].ToSingle() };
					var translateParams = new List<int> { T[0, 2].ToInt32(), T[1, 2].ToInt32() };

					outTensor = torchvision.transforms.functional.affine(im, shearParams, a, translateParams, s);
					outTensor = torchvision.transforms.functional.crop(outTensor, (int)im.shape[1] - imageSize, (int)im.shape[2] - imageSize, imageSize, imageSize).contiguous();
				}
			}

			long n = targets.shape[0];
			if (n > 0)
			{
				Tensor newT = torch.zeros([n, 4]).to(device);
				Tensor xy = torch.ones([n * 4, 3]).to(device);
				xy[TensorIndex.Ellipsis, 0..2] = targets.index_select(1, torch.tensor(new long[] { 1, 2, 3, 4, 1, 4, 3, 2 }).to(device)).reshape(n * 4, 2).to(device);  // x1y1, x2y2, x1y2, x2y1
				xy = xy.mm(M.T);
				xy = perspective == 0 ? xy[TensorIndex.Ellipsis, 0..2].reshape(n, 8) : (xy[TensorIndex.Ellipsis, 0..2] / xy[TensorIndex.Ellipsis, 2..3]);
				Tensor x = xy.index_select(1, torch.tensor(new long[] { 0, 2, 4, 6 }).to(device));
				Tensor y = xy.index_select(1, torch.tensor(new long[] { 1, 3, 5, 7 }).to(device));
				newT = torch.concatenate([x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values]).reshape(4, n).T;

				newT = newT.index_put_(newT.index_select(1, torch.tensor(new long[] { 0, 2 }).to(device)).clip(0, imageSize), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(0, 3, 2) });
				newT = newT.index_put_(newT.index_select(1, torch.tensor(new long[] { 1, 3 }).to(device)).clip(0, imageSize), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(1, 4, 2) });

				Tensor idx = box_candidates(box1: targets[TensorIndex.Ellipsis, 1..5].T * s, box2: newT.T, area_thr: 0.1f);
				targets = targets[idx];
				targets[TensorIndex.Ellipsis, 1..5] = newT[idx];
			}

			return (outTensor.contiguous(), targets);
		}

		public static torch.Tensor GetRotationMatrix2D(float angle, float scale)
		{
			// 将角度转换为弧度
			float theta = angle * (float)Math.PI / 180.0f;

			// 计算旋转矩阵的元素
			float cosTheta = (float)Math.Cos(theta);
			float sinTheta = (float)Math.Sin(theta);

			// 创建旋转矩阵
			var R = torch.tensor(new float[,]
			{
				{ scale * cosTheta, -scale * sinTheta, 0 },
				{ scale * sinTheta, scale * cosTheta, 0 },
				{ 0, 0, 1 }
			});
			return R;
		}

		private Tensor box_candidates(Tensor box1, Tensor box2, float wh_thr = 2, float ar_thr = 100, float area_thr = 0.1f, double eps = 1e-16)
		{
			/*
			Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold
			`ar_thr`, and area ratio threshold `area_thr`.
			box1(4,n) is before augmentation, box2(4,n) is after augmentation.
			*/

			var (w1, h1) = (box1[2] - box1[0], box1[3] - box1[1]);
			var (w2, h2) = (box2[2] - box2[0], box2[3] - box2[1]);
			var ar = torch.maximum(w2 / (h2 + eps), h2 / (w2 + eps)); // aspect ratio
			return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr); // candidates
		}

	}
}

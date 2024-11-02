using OpenCvSharp;
using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using TorchSharp;
using static OpenCvSharp.FileStorage;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace Yolov5
{
	internal class YoloDataset : torch.utils.data.Dataset
	{
		private static double[] means = [0.485, 0.456, 0.406], stdevs = [0.229, 0.224, 0.225];
		private string rootPath = string.Empty;
		private int resizeWidth = 0;
		private int resizeHeight = 0;
		private List<string> imageFiles = new List<string>();

		public YoloDataset(string rootPath, int resizeWidth = 640, int resizeHeight = 640)
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
			this.resizeWidth = resizeWidth;
			this.resizeHeight = resizeHeight;
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
			string file = imageFiles[(int)index];
			Tensor orgImageTensor = torchvision.io.read_image(file);
			var (imgTensor, scale, pad) = Letterbox(orgImageTensor, resizeWidth, resizeHeight);
			imgTensor = torchvision.transforms.functional.normalize(imgTensor.unsqueeze(0) / 255.0f, means, stdevs).squeeze(0);
			//imgTensor = imgTensor / 255.0f;
			outputs.Add("image", imgTensor);
			outputs.Add("index", torch.tensor(index));
			return outputs;
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

		public Tensor GetLabelTensor(long index)
		{
			Tensor orgImageTensor = torchvision.io.read_image(imageFiles[(int)index]);
			var (imgTensor, scale, pad) = Letterbox(orgImageTensor, resizeWidth, resizeHeight);
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

					labelArray[i, 2] = (float.Parse(labels[2]) * (resizeHeight - 2 * pad) + pad) / (float)resizeHeight;
					labelArray[i, 4] = float.Parse(labels[4]) * (resizeHeight - 2 * pad) / (float)resizeHeight;
				}
				else
				{
					labelArray[i, 1] = (float.Parse(labels[1]) * (resizeWidth - 2 * pad) + pad) / (float)resizeWidth;
					labelArray[i, 3] = float.Parse(labels[3]) * (resizeWidth - 2 * pad) / (float)resizeWidth;

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

		public void load_mosaic(long index)
		{
			int s = 640;
			int[] mosaic_border = [-320, -320];
			Int64[] indexs = Sample(index, 0, (int)Count, 4);
			Random random = new Random();
			int xc = random.Next(-mosaic_border[0], 2 * s + mosaic_border[0]);
			int yc = random.Next(-mosaic_border[1], 2 * s + mosaic_border[1]);
			indexs = [1, 10, 5, 9];
			yc = 441;
			xc = 439;
			var img4 = torch.full([3, s * 2, s * 2], 114, ScalarType.Byte); // base image with 4 tiles
			List<Tensor> label4 = new List<Tensor>();
			for (int i = 0; i < 4; i++)
			{
				int x1a = 0, y1a = 0, x2a = 0, y2a = 0, x1b = 0, y1b = 0, x2b = 0, y2b = 0;
				Tensor img = GetOrgImage(indexs[i]);
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
					(x1a, y1a, x2a, y2a) = (xc, Math.Max(yc - h, 0), Math.Min(xc + w, s * 2), yc);
					(x1b, y1b, x2b, y2b) = (0, h - (y2a - y1a), Math.Min(w, x2a - x1a), h);
				}
				else if (i == 2)  // bottom left
				{
					(x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), yc, xc, Math.Min(s * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (w - (x2a - x1a), 0, w, Math.Min(y2a - y1a, h));
				}
				else if (i == 3) // bottom right
				{
					(x1a, y1a, x2a, y2a) = (xc, yc, Math.Min(xc + w, s * 2), Math.Min(s * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (0, 0, Math.Min(w, x2a - x1a), Math.Min(y2a - y1a, h));
				}
				img4[0..3, y1a..y2a, x1a..x2a] = img[0..3, y1b..y2b, x1b..x2b];

				int padw = x1a - x1b;
				int padh = y1a - y1b;

				Tensor labels = GetOrgLabelTensor(indexs[i]);
				labels[TensorIndex.Ellipsis, 1..5] = xywhn2xyxy(labels[TensorIndex.Ellipsis, 1..5], w, h, padw, padh);
				label4.Add(labels);
			}
			var labels4 = torch.concat(label4, 0);
			labels4[TensorIndex.Ellipsis, 1..5] = labels4[TensorIndex.Ellipsis, 1..5].clip(0, 2 * s);

			var (im, targets) = random_perspective(img4, labels4, degrees: 0, translate: 0.1f, scale: 0.5f, shear: 0, perspective: 0.0f, mosaic_border[0], mosaic_border[1]);

			targets[TensorIndex.Ellipsis, 1..5] = xyxy2xywhn(targets[TensorIndex.Ellipsis, 1..5], w: (int)im.shape[1], h: (int)im.shape[2], clip: true, eps: 1e-3f);


			Bitmap bitmap = new Bitmap(640, 640);
			for (int i = 0; i < bitmap.Width; i++)
			{
				for (int j = 0; j < bitmap.Height; j++)
				{
					int b = im[2][j][i].ToInt32();
					int g = im[1][j][i].ToInt32();
					int r = im[0][j][i].ToInt32();
					Color c = Color.FromArgb(r, g, b);
					bitmap.SetPixel(i, j, c);
				}
			}

			Graphics gg = Graphics.FromImage(bitmap);
			for (int i = 0; i < targets.shape[0]; i++)
			{
				int xxc = (int)(targets[i][1].ToSingle() * 640);
				int yyc = (int)(targets[i][2].ToSingle() * 640);
				int ww = (int)(targets[i][3].ToSingle() * 640);
				int hh = (int)(targets[i][4].ToSingle() * 640);
				Rectangle rectangle = new Rectangle((int)(xxc - ww / 2), (int)(yyc - hh / 2), ww, hh);
				gg.DrawRectangle(Pens.Red, rectangle);
				gg.Save();
			}

			bitmap.Save("bitmap.jpg");

			//torchvision.io.write_jpeg(im, "im.jpg");
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
				int number = random.Next(min, max + 1);
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
			int height = (int)im.shape[1] + borderY * 2;
			int width = (int)im.shape[2] + borderX * 2;

			// Center
			Tensor C = torch.eye(3);
			C[0, 2] = -im.shape[2] / 2; // x translation (pixels)
			C[1, 2] = -im.shape[1] / 2; // y translation (pixels)

			// Perspective
			Tensor P = torch.eye(3);
			P[2, 0] = torch.rand(1).ToSingle() * 2 * perspective - perspective;   // x perspective (about y)
			P[2, 1] = torch.rand(1).ToSingle() * 2 * perspective - perspective;   // y perspective (about x)

			// Rotation and Scale
			float a = torch.rand(1).ToSingle() * 2 * degrees - degrees;
			float s = 1 + scale - torch.rand(1).ToSingle() * 2 * scale;
			s = 1.398173121357879f;
			Tensor R = GetRotationMatrix2D(angle: a, scale: s);

			// Shear
			Tensor S = torch.eye(3);
			S[0, 1] = Math.Tan((torch.rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // x shear (deg)
			S[1, 0] = Math.Tan((torch.rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // y shear (deg)

			// Translation
			Tensor T = torch.eye(3);
			T[0, 2] = (0.5f + translate - torch.rand(1).ToSingle() * 2 * translate) * width;    // x translation(pixels)
			T[1, 2] = (0.5f + translate - torch.rand(1).ToSingle() * 2 * translate) * height;   // y translation(pixels)

			var M = T.mm(S).mm(R).mm(P).mm(C);

			Tensor outTensor = torch.zeros([width, height, 3], ScalarType.Byte);
			float[,] fffff = {
				{1.3982f, 0, -588.74f },
				{ 0, 1.3982f, -548.55f },
				{ 0, 0, 1.0f }};

			M = torch.tensor(fffff);
			Mat ommm = new Mat();
			if (borderX != 0 || borderY != 0 || M.bytes != torch.eye(3).bytes)
			{
				if (perspective != 0)
				{
					//im = WarpPerspective(im, M, width, height, (114, 114, 114));
				}
				else
				{
					// 提取仿射变换的参数
					var affineParams = M[TensorIndex.Slice(0, 2)].view(2, 3).tolist();
					var shearParams = new List<float> { S[0, 1].ToSingle(), S[1, 0].ToSingle() };
					var translateParams = new List<int> { T[0, 2].ToInt32(), T[1, 2].ToInt32() };
					OpenCvSharp.Mat orgMat = new OpenCvSharp.Mat(1280, 1280, OpenCvSharp.MatType.CV_8UC3);
					byte[] bt = im.permute(2, 1, 0).data<byte>().ToArray();
					Marshal.Copy(bt, 0, orgMat.Data, bt.Length);

					float[] mc = M[0..2].data<float>().ToArray();
					OpenCvSharp.Mat mMat = new OpenCvSharp.Mat(2, 3, OpenCvSharp.MatType.CV_32F);
					Marshal.Copy(mc, 0, mMat.Data, mc.Length);
					OpenCvSharp.Mat outMat = orgMat.WarpAffine(mMat, dsize: new OpenCvSharp.Size(width, height), borderValue: new OpenCvSharp.Scalar(114, 114, 114));
					byte[] bbbb = new byte[640 * 640 * 3];
					Marshal.Copy(outMat.Data, bbbb, 0, bbbb.Length);
					outTensor.bytes = bbbb;
					outTensor = outTensor.permute(2, 1, 0);
				}
			}

			long n = targets.shape[0];
			if (n > 0)
			{
				Tensor newT = torch.zeros([n, 4]);
				Tensor xy = torch.ones([n * 4, 3]);
				xy[TensorIndex.Ellipsis, 0..2] = targets.index_select(1, new long[] { 1, 2, 3, 4, 1, 4, 3, 2 }).reshape(n * 4, 2);  // x1y1, x2y2, x1y2, x2y1
				xy = xy.mm(M.T);
				xy = perspective == 0 ? xy[TensorIndex.Ellipsis, 0..2].reshape(n, 8) : (xy[TensorIndex.Ellipsis, 0..2] / xy[TensorIndex.Ellipsis, 2..3]);
				Tensor x = xy.index_select(1, new long[] { 0, 2, 4, 6 });
				Tensor y = xy.index_select(1, new long[] { 1, 3, 5, 7 });
				newT = torch.concatenate([x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values]).reshape(4, n).T;

				newT = newT.index_put_(newT.index_select(1, new long[] { 0, 2 }).clip(0, width), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(0, 3, 2) });
				newT = newT.index_put_(newT.index_select(1, new long[] { 1, 3 }).clip(0, width), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(1, 4, 2) });

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


		//public torch.Tensor WarpPerspective(torch.Tensor im, torch.Tensor M, int width, int height, (int, int, int) borderValue)
		//{
		//	// 将图像转换为浮点数类型
		//	im = im.to(ScalarType.Float32);

		//	// 创建填充颜色张量
		//	var fill = torch.tensor(new float[] { borderValue.Item1, borderValue.Item2, borderValue.Item3 }, ScalarType.Float32).view(1, 3, 1, 1);

		//	// 应用透视变换
		//	var transformedIm = torchvision.transforms.functional.perspective(im, M, new long[] { width, height }, fill: fill);

		//	return transformedIm;
		//}

	}
}

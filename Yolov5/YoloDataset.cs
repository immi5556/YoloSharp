using TorchSharp;
using static TorchSharp.torch;

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
	}
}

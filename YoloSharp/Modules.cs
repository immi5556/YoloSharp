using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	public class Modules
	{
		public class Conv : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly BatchNorm2d bn;
			private readonly bool act;

			public Conv(int in_channels, int out_channels, int kernel_size, int stride = 1, int? padding = null, int groups = 1, bool bias = true, bool act = true) : base("Conv")
			{
				if (padding == null)
				{
					padding = (kernel_size) / 2;
				}

				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding.Value, groups: groups, bias: bias);
				bn = BatchNorm2d(out_channels);
				this.act = act;
				RegisterComponents();
			}

			public Conv(int in_channels, int out_channels, (long, long) kernel_size, (long, long) stride, (long, long)? padding = null, int groups = 1, bool bias = true, bool act = true) : base("Conv")
			{
				if (padding == null)
				{
					padding = (kernel_size.Item1 / 2, kernel_size.Item2 / 2);
				}

				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding.Value);
				bn = BatchNorm2d(out_channels);
				this.act = act;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				var x = conv.forward(input);
				Module<Tensor, Tensor> ac = act ? SiLU() : Identity();
				return ac.forward(bn.forward(x));
			}
		}

		public class Bottleneck : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			bool add;

			public Bottleneck(int inChannels, int outChannels, (int, int) kernal, bool shortcut = true, int groups = 1, float e = 0.5f) : base("bottleneck")
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, kernal.Item1, 1);
				cv2 = new Conv(c, outChannels, kernal.Item2, 1, groups: groups);
				add = shortcut && inChannels == outChannels;
				RegisterComponents();
			}

			public Bottleneck(int inChannels, int outChannels, ((int, int), (int, int)) kernal, bool shortcut = true, int groups = 1, float e = 0.5f) : base("bottleneck")
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, kernal.Item1, (1, 1));
				cv2 = new Conv(c, outChannels, kernal.Item2, (1, 1), groups: groups);
				add = shortcut && inChannels == outChannels;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return add ? input + cv2.forward(cv1.forward(input)) : cv2.forward(cv1.forward(input));
			}
		}

		public class C3 : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Conv cv3;
			private readonly Sequential m = Sequential();

			public C3(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f) : base("C3")
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, 1, 1);
				cv2 = new Conv(inChannels, c, 1, 1);
				cv3 = new Conv(2 * c, outChannels, 1);

				for (int i = 0; i < n; i++)
				{
					m = m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, e: 1.0f));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return cv3.forward(cat([m.forward(cv1.forward(input)), cv2.forward(input)], 1));
			}
		}

		public class C2f : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Sequential m = Sequential();
			public C2f(int inChannels, int outChannels, int n = 1, bool shortcut = false, int groups = 1, float e = 0.5f) : base("C2f")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, 2 * c, 1, 1);
				this.cv2 = new Conv((2 + n) * c, outChannels, 1);  // optional act=FReLU(c2)
				for (int i = 0; i < n; i++)
				{
					m = m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, e));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				var y = this.cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					 y.Add(m[i].call(y.Last()));
				}
				return cv2.forward(cat(y, 1));
			}
		}

		public class SPPF : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly MaxPool2d m;

			public SPPF(int inChannels, int outChannels, int kernalSize = 5) : base("SPPF")
			{
				int c = inChannels / 2;
				cv1 = new Conv(inChannels, c, 1, 1);
				cv2 = new Conv(c * 4, outChannels, 1, 1);
				m = MaxPool2d(kernelSize: kernalSize, stride: 1, padding: kernalSize / 2);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				var x = cv1.forward(input);
				var y1 = m.forward(x);
				var y2 = m.forward(y1);

				return cv2.forward(cat(new[] { x, y1, y2, m.forward(y2) }, 1));
			}
		}

		public class Detect : Module<Tensor[], Tensor[]>
		{
			Tensor stride;  //strides computed during build
			bool dynamic = false;  // force grid reconstruction
			bool export = false;// export mode

			int[] model_stride = [8, 16, 32];

			private readonly int nc;
			private readonly int no;
			private readonly int nl;
			private readonly int na;
			private List<Tensor> grid; // 存储网格坐标的列表
			private List<Tensor> anchor_grid;
			private Tensor anchors;
			private readonly Sequential m = Sequential();

			public Detect(int nc, int[] ch, float[][] anchors, bool inplace = true) : base("Detect")
			{
				this.nc = nc;
				no = nc + 5;// =85 每个类别需添加位置与置信度
				nl = anchors.Length;
				na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				float[] an = new float[anchors.Length * anchors[0].Length];
				for (int i = 0; i < anchors.Length; i++)
				{
					for (int j = 0; j < anchors[1].Length; j++)
					{
						an[i * anchors[0].Length + j] = anchors[i][j];
					}
				}
				this.anchors = tensor(an, [anchors.Length, anchors[0].Length / 2, 2]);
				stride = tensor(model_stride, ScalarType.Float32);
				grid = new List<Tensor>(nl);
				anchor_grid = new List<Tensor>(nl);
				for (int i = 0; i < nl; i++)
				{
					grid.Add(empty(0));
					anchor_grid.Add(empty(0));
				}
				for (int i = 0; i < ch.Length; i++)
				{
					m = m.append(Conv2d(ch[i], no * na, 1));
				}
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				List<Tensor> z = new List<Tensor>();
				for (int i = 0; i < nl; i++)
				{
					x[i] = ((Module<Tensor, Tensor>)m[i]).forward(x[i]);
					long bs = x[i].shape[0];
					int ny = (int)x[i].shape[2];
					int nx = (int)x[i].shape[3];
					x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
					if (!training)
					{
						(grid[i], anchor_grid[i]) = _make_grid(nx, ny, i);
						Tensor[] re = x[i].sigmoid().split([2, 2, nc + 1], 4);
						Tensor xy = re[0];
						Tensor wh = re[1];
						Tensor conf = re[2];

						xy = (xy * 2 + grid[i]) * stride[i];  // xy
						wh = (wh * 2).pow(2) * anchor_grid[i];  // wh
						Tensor y = cat([xy, wh, conf], 4);
						z.Add(y.view(bs, na * nx * ny, no));
					}
				}

				if (training)
				{
					return x;
				}
				else
				{
					var list = new List<Tensor>() { cat(z, 1) };
					list.AddRange(x);
					return list.ToArray();
				}
			}

			private (Tensor, Tensor) _make_grid(int nx = 20, int ny = 20, int i = 0)
			{
				var d = anchors[i].device;
				var t = anchors[i].dtype;

				long[] shape = [1, na, ny, nx, 2]; // grid shape
				Tensor y = arange(ny, t, d);
				Tensor x = arange(nx, t, d);
				Tensor[] xy = meshgrid([y, x], indexing: "ij");
				Tensor yv = xy[0];
				Tensor xv = xy[1];
				Tensor grid = stack([xv, yv], 2).expand(shape) - 0.5f;  // add grid offset, i.e. y = 2.0 * x - 0.5

				Tensor anchor_grid = (anchors[i] * stride[i]).view([1, na, 1, 1, 2]).expand(shape);

				return (grid, anchor_grid);
			}
		}




	}
}

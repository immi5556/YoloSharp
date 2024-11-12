using TorchSharp;
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
			public Sequential m = Sequential();

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

		public class C3k : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Conv cv3;
			public Sequential m = Sequential();
			public C3k(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f) : base("C3k")
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, 1, 1);
				cv2 = new Conv(inChannels, c, 1, 1);
				cv3 = new Conv(2 * c, outChannels, 1);
				for (int i = 0; i < n; i++)
				{
					this.m = this.m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, e: 1.0f));
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
			public Sequential m = Sequential();
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

		public class C3k2 : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			public Sequential m = Sequential();
			public C3k2(int inChannels, int outChannels, int n = 1, bool c3k = false, bool shortcut = false, int groups = 1, float e = 0.5f, int k = 3) : base("C3k2")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, 2 * c, 1, 1);
				this.cv2 = new Conv((2 + n) * c, outChannels, 1);  // optional act=FReLU(c2)
				for (int i = 0; i < n; i++)
				{
					if (c3k)
					{
						this.m = this.m.append(new C3k(c, c, 2, shortcut, groups, e));
					}
					else
					{

						this.m = this.m.append(new Bottleneck(c, c, (k, k), shortcut, groups, e));
					}
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

		public class C2PSA : Module<Tensor, Tensor>
		{
			private readonly int c;
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Sequential m = Sequential();

			public C2PSA(int inChannel, int outChannel, int n = 1, float e = 0.5f) : base("C2PSA")
			{
				if (inChannel != outChannel)
				{
					throw new ArgumentException("in channel not equals to out channel");
				}
				this.c = (int)(inChannel * e);
				this.cv1 = new Conv(inChannel, 2 * c, 1, 1);
				this.cv2 = new Conv(2 * c, outChannel, 1);

				for (int i = 0; i < n; i++)
				{
					m = m.append(new PSABlock(c, attn_ratio: 0.5f, num_heads: c / 64));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				Tensor[] ab = this.cv1.forward(x).split([this.c, this.c], dim: 1);
				Tensor a = ab[0];
				Tensor b = ab[1];
				b = this.m.forward(b);
				return this.cv2.forward(torch.cat([a, b], 1));
			}
		}

		public class PSABlock : Module<Tensor, Tensor>
		{
			private readonly Attention attn;
			private readonly Sequential ffn;
			private readonly bool add;

			public PSABlock(int c, float attn_ratio = 0.5f, int num_heads = 4, bool shortcut = true) : base("PSABlock")
			{
				this.attn = new Attention(c, attn_ratio: attn_ratio, num_heads: num_heads);
				this.ffn = nn.Sequential(new Conv(c, c * 2, 1), new Conv(c * 2, c, 1, act: false));
				this.add = shortcut;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				x = this.add ? (x + this.attn.forward(x)) : this.attn.forward(x);
				x = this.add ? (x + this.ffn.forward(x)) : this.ffn.forward(x);
				return x;
			}
		}

		public class Attention : Module<Tensor, Tensor>
		{
			private int num_heads;
			private int head_dim;
			private int key_dim;
			private float scale;

			private readonly Conv qkv;
			private readonly Conv proj;
			private readonly Conv pe;

			public Attention(int dim, int num_heads = 8, float attn_ratio = 0.5f) : base("Attention")
			{
				this.num_heads = num_heads;
				this.head_dim = dim / num_heads;
				this.key_dim = (int)(this.head_dim * attn_ratio);
				this.scale = (float)Math.Pow(key_dim, -0.5);

				int nh_kd = this.key_dim * num_heads;
				int h = dim + nh_kd * 2;

				this.qkv = new Conv(dim, h, 1, act: false);
				this.proj = new Conv(dim, dim, 1, act: false);
				this.pe = new Conv(dim, dim, 3, 1, groups: dim, act: false);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				long B = x.shape[0];
				long C = x.shape[1];
				long H = x.shape[2];
				long W = x.shape[3];

				long N = H * W;

				Tensor qkv = this.qkv.forward(x);

				Tensor[] qkv_mix = qkv.view(B, this.num_heads, this.key_dim * 2 + this.head_dim, N).split(
					[this.key_dim, this.key_dim, this.head_dim], dim: 2
				);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];

				Tensor attn = q.transpose(-2, -1).matmul(k) * this.scale;
				attn = attn.softmax(dim: -1);
				x = (v.matmul(attn.transpose(-2, -1))).view(B, C, H, W) + this.pe.forward(v.reshape(B, C, H, W));
				x = this.proj.forward(x);
				return x;

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

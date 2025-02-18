using System.Numerics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	internal class Modules
	{
		public class Conv : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly BatchNorm2d bn;
			private readonly bool act;
			private double eps = 0.001;
			private double momentum = 0.03;

			public Conv(int in_channels, int out_channels, int kernel_size, int stride = 1, int? padding = null, int groups = 1, int d = 1, bool bias = false, bool act = true) : base("Conv")
			{
				if (padding == null)
				{
					padding = (kernel_size) / 2;
				}

				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding.Value, groups: groups, bias: bias, dilation: d);
				bn = BatchNorm2d(out_channels, eps: eps, momentum: momentum);
				this.act = act;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				Module<Tensor, Tensor> ac = act ? SiLU(true) : Identity();
				Tensor result = ac.forward(bn.forward(conv.forward(input)));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class DWConv : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly BatchNorm2d bn;
			private readonly bool act;

			public DWConv(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int d = 1, bool act = true, bool bias = false) : base("DWConv")
			{
				int groups = (int)BigInteger.GreatestCommonDivisor(in_channels, out_channels);
				int padding = (kernel_size) / 2;
				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups: groups, dilation: d, bias: bias);
				bn = BatchNorm2d(out_channels);
				this.act = act;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var x = conv.forward(input);
				Module<Tensor, Tensor> ac = act ? SiLU() : Identity();
				Tensor result = ac.forward(bn.forward(x));
				return result.MoveToOuterDisposeScope();
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

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				Tensor result = add ? input + cv2.forward(cv1.forward(input)) : cv2.forward(cv1.forward(input));
				return result.MoveToOuterDisposeScope();
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
					m = m.append(new Bottleneck(c, c, (1, 3), shortcut, groups, e: 1.0f));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				Tensor result = cv3.forward(cat([m.forward(cv1.forward(input)), cv2.forward(input)], 1));
				return result.MoveToOuterDisposeScope();
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
				using var _ = NewDisposeScope();
				Tensor result = cv3.forward(cat([m.forward(cv1.forward(input)), cv2.forward(input)], 1));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class C2f : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			public readonly int c;
			public Sequential m = Sequential();
			public C2f(int inChannels, int outChannels, int n = 1, bool shortcut = false, int groups = 1, float e = 0.5f) : base("C2f")
			{
				this.c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, 2 * c, 1, 1);
				this.cv2 = new Conv((2 + n) * c, outChannels, 1);  // optional act=FReLU(outChannels)
				for (int i = 0; i < n; i++)
				{
					m = m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, 1));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var y = this.cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					y.Add(m[i].call(y.Last()));
				}
				Tensor result = cv2.forward(cat(y, 1));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class C3k2 : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			public Sequential m = Sequential();
			public C3k2(int inChannels, int outChannels, int n = 1, bool c3k = false, float e = 0.5f, int groups = 1, bool shortcut = true) : base("C3k2")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, 2 * c, 1, 1);
				this.cv2 = new Conv((2 + n) * c, outChannels, 1);  // optional act=FReLU(outChannels)
				for (int i = 0; i < n; i++)
				{
					if (c3k)
					{
						this.m = this.m.append(new C3k(c, c, 2, shortcut, groups));
					}
					else
					{
						this.m = this.m.append(new Bottleneck(c, c, (3, 3), shortcut, groups));
					}
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var y = this.cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					y.Add(m[i].call(y.Last()));
				}
				Tensor result = cv2.forward(cat(y, 1));
				return result.MoveToOuterDisposeScope();
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
				m = MaxPool2d(kernalSize, stride: 1, padding: kernalSize / 2);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var x = cv1.forward(input);
				var y1 = m.forward(x);
				var y2 = m.forward(y1);
				Tensor result = cv2.forward(cat(new[] { x, y1, y2, m.forward(y2) }, 1));
				return result.MoveToOuterDisposeScope();
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
				using var _ = NewDisposeScope();
				Tensor[] ab = this.cv1.forward(x).split([this.c, this.c], dim: 1);
				Tensor a = ab[0];
				Tensor b = ab[1];
				b = this.m.forward(b);
				Tensor result = this.cv2.forward(torch.cat([a, b], 1));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class PSABlock : Module<Tensor, Tensor>
		{
			private readonly Attention attn; // can use ScaledDotProductAttention instead
			private readonly Sequential ffn;
			private readonly bool add;

			public PSABlock(int c, float attn_ratio = 0.5f, int num_heads = 4, bool shortcut = true) : base("PSABlock")
			{
				this.attn = new Attention(c); 
				this.ffn = nn.Sequential(new Conv(c, c * 2, 1), new Conv(c * 2, c, 1, act: false));
				this.add = shortcut;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				x = this.add ? (x + this.attn.forward(x)) : this.attn.forward(x);
				x = this.add ? (x + this.ffn.forward(x)) : this.ffn.forward(x);
				return x.MoveToOuterDisposeScope();
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
				using var _ = NewDisposeScope();
				long B = x.shape[0];
				long C = x.shape[1];
				long H = x.shape[2];
				long W = x.shape[3];

				long N = H * W;

				Tensor qkv = this.qkv.forward(x);

				Tensor[] qkv_mix = qkv.view(B, this.num_heads, this.key_dim * 2 + this.head_dim, N).split([this.key_dim, this.key_dim, this.head_dim], dim: 2);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];

				Tensor attn = q.transpose(-2, -1).matmul(k) * this.scale;
				attn = attn.softmax(dim: -1);
				x = (v.matmul(attn.transpose(-2, -1))).view(B, C, H, W) + this.pe.forward(v.reshape(B, C, H, W));
				x = this.proj.forward(x);

				return x.MoveToOuterDisposeScope();

			}
		}


		public class ScaledDotProductAttention : Module<Tensor, Tensor>
		{
			private int num_heads;
			private int head_dim;
			private int key_dim;

			private readonly Conv qkv;
			private readonly Conv proj;
			private readonly Conv pe;

			public ScaledDotProductAttention(int dim, int num_heads = 8, float attn_ratio = 0.5f) : base("Attention")
			{
				this.num_heads = num_heads;
				this.head_dim = dim / num_heads;
				this.key_dim = (int)(this.head_dim * attn_ratio);

				int nh_kd = this.key_dim * num_heads;
				int h = dim + nh_kd * 2;

				this.qkv = new Conv(dim, h, 1, act: false);
				this.proj = new Conv(dim, dim, 1, act: false);
				this.pe = new Conv(dim, dim, 3, 1, groups: dim, act: false);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				long B = x.shape[0];
				long C = x.shape[1];
				long H = x.shape[2];
				long W = x.shape[3];

				long N = H * W;

				Tensor qkv = this.qkv.forward(x);

				Tensor[] qkv_mix = qkv.view(B, this.num_heads, this.key_dim * 2 + this.head_dim, N).split([this.key_dim, this.key_dim, this.head_dim], dim: 2);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];

				q = q.transpose(-2, -1); // [B, num_heads, N, key_dim]
				k = k.transpose(-2, -1); // [B, num_heads, N, key_dim]
				v = v.transpose(-2, -1); // [B, num_heads, N, head_dim]

				Tensor attn_output = functional.scaled_dot_product_attention(q, k, v);

				attn_output = attn_output.transpose(-2, -1); // [B, num_heads, N, head_dim]
				attn_output = attn_output.contiguous();

				if (B * this.num_heads * N * this.head_dim != B * C * H * W)
				{
					throw new InvalidOperationException("Shape mismatch: Cannot reshape attn_output to [B, C, H, W].");
				}

				attn_output = attn_output.view(B, C, H, W);
				x = attn_output + this.pe.forward(v.reshape(B, C, H, W));
				x = this.proj.forward(x);

				return x.MoveToOuterDisposeScope();
			}
		}

		public class SCDown : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			public SCDown(int inChannel, int outChannel, int k, int s) : base("SCDown")
			{
				this.cv1 = new Conv(inChannel, outChannel, 1, 1);
				this.cv2 = new Conv(outChannel, outChannel, kernel_size: k, stride: s, groups: outChannel, act: false);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor result = this.cv2.forward(this.cv1.forward(x));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class C2fCIB : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			public Sequential m = Sequential();
			public C2fCIB(int inChannels, int outChannels, int n = 1, bool shortcut = false, bool lk = false, int g = 1, float e = 0.5f) : base("C2fCIB")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, 2 * c, 1, 1);
				this.cv2 = new Conv((2 + n) * c, outChannels, 1);  // optional act=FReLU(outChannels)
				for (int i = 0; i < n; i++)
				{
					m = m.append(new CIB(c, c, shortcut, e: 1.0f, lk: lk));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var y = this.cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					y.Add(m[i].call(y.Last()));
				}
				Tensor result = cv2.forward(cat(y, 1));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class CIB : Module<Tensor, Tensor>
		{
			Sequential cv1;
			bool add;
			public CIB(int inChannels, int outChannels, bool shortcut = true, float e = 0.5f, bool lk = false) : base("CIB")
			{
				int c = (int)(outChannels * e);  // hidden channels
				this.cv1 = nn.Sequential(
					new Conv(inChannels, inChannels, 3, groups: inChannels),
					new Conv(inChannels, 2 * c, 1),
					lk ? new RepVGGDW(2 * c) : new Conv(2 * c, 2 * c, 3, groups: 2 * c),
					new Conv(2 * c, outChannels, 1),
					new Conv(outChannels, outChannels, 3, groups: outChannels));
				this.add = shortcut && (inChannels == outChannels);

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor result = this.add ? (x + this.cv1.forward(x)) : this.cv1.forward(x);
				return result.MoveToOuterDisposeScope();
			}
		}

		public class RepVGGDW : Module<Tensor, Tensor>
		{
			private readonly Conv conv;
			private readonly Conv conv1;
			private readonly int dim;
			private readonly Module<Tensor, Tensor> act;
			public RepVGGDW(int ed) : base("RepVGGDW")
			{
				this.conv = new Conv(ed, ed, 7, 1, 3, groups: ed, act: false);
				this.conv1 = new Conv(ed, ed, 3, 1, 1, groups: ed, act: false);
				this.dim = ed;
				this.act = nn.SiLU();

				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				Tensor result = this.act.forward(this.conv.forward(x) + this.conv1.forward(x));
				return result.MoveToOuterDisposeScope();
			}
		}

		public class DFL : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly int c1;
			public DFL(int c1 = 16) : base("DFL")
			{
				this.conv = nn.Conv2d(c1, 1, 1, bias: false);
				Tensor x = torch.arange(c1, dtype: torch.float32);
				this.conv.weight = nn.Parameter(x.view(1, c1, 1, 1));
				this.c1 = c1;

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				long b = x.shape[0];  // batch, channels, anchors
				long a = x.shape[2];

				Tensor result = this.conv.forward(x.view(b, 4, this.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a);
				return result.MoveToOuterDisposeScope();
			}
		}

		public class Concat : Module<Tensor[], Tensor>
		{
			private readonly int dim;
			public Concat(int dim = 1) : base("Concat")
			{
				this.dim = dim;
			}

			public override Tensor forward(Tensor[] input)
			{
				using var _ = NewDisposeScope();
				Tensor result = torch.concat(input, dim: dim);
				return result.MoveToOuterDisposeScope();
			}
		}

		public class Yolov5Detect : Module<Tensor[], Tensor[]>
		{
			bool dynamic = false;  // force grid reconstruction
			bool export = false;// export mode

			private readonly int nc;
			private readonly int no;
			private readonly int nl;
			private readonly int na;
			//private List<Tensor> grid; // 存储网格坐标的列表

			//private readonly Tensor anchors;
			private readonly Sequential m = Sequential();
			private float[][] anchors;
			private readonly int[] ch;

			private torch.Device device;
			private torch.ScalarType scalarType;

			public Yolov5Detect(int nc, int[] ch, float[][] anchors, bool inplace = true) : base("Yolov5Detect")
			{
				this.nc = nc;
				no = nc + 5;// =85 每个类别需添加位置与置信度
				nl = anchors.Length;
				na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				this.anchors = anchors;
				this.ch = ch;
				//grid = new List<Tensor>(nl);

				for (int i = 0; i < ch.Length; i++)
				{
					m = m.append(Conv2d(ch[i], no * na, 1));
				}
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				this.device = x[0].device;
				this.scalarType = x[0].dtype;

				List<Tensor> z = new List<Tensor>();
				Tensor stride = tensor(new int[] { 8, 16, 32 }, dtype: scalarType, device: device);  //strides computed during build
				for (int i = 0; i < nl; i++)
				{
					x[i] = ((Module<Tensor, Tensor>)m[i]).forward(x[i]);
					long bs = x[i].shape[0];
					int ny = (int)x[i].shape[2];
					int nx = (int)x[i].shape[3];
					x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
					if (!training)
					{
						var (grid, anchor_grid) = _make_grid(nx, ny, i);
						Tensor[] re = x[i].sigmoid().split([2, 2, nc + 1], 4);
						Tensor xy = re[0];
						Tensor wh = re[1];
						Tensor conf = re[2];

						xy = (xy * 2 + grid) * stride[i];  // xy
						wh = (wh * 2).pow(2) * anchor_grid;  // wh
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
					return list.ToArray();
				}
			}

			private (Tensor, Tensor) _make_grid(int nx = 20, int ny = 20, int i = 0)
			{
				using var _ = NewDisposeScope();
				float[] an = new float[this.anchors.Length * this.anchors[0].Length];
				for (int ii = 0; ii < this.anchors.Length; ii++)
				{
					for (int j = 0; j < this.anchors[1].Length; j++)
					{
						an[ii * this.anchors[0].Length + j] = this.anchors[ii][j];
					}
				}
				Tensor anchors = tensor(an, [this.anchors.Length, this.anchors[0].Length / 2, 2], dtype: scalarType, device: device);
				Tensor stride = tensor(new int[] { 8, 16, 32 }, dtype: scalarType, device: device);  //strides computed during build
																									 //Tensor stride = tensor(ch, dtype: scalarType, device: device) / 8;  //strides computed during build
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

				return (grid.MoveToOuterDisposeScope(), anchor_grid.MoveToOuterDisposeScope());
			}
		}

		public class YolovDetect : Module<Tensor[], Tensor[]>
		{
			private int max_det = 300; // max_det
			private long[] shape = null;
			private Tensor anchors = torch.empty(0); // init
			private Tensor strides = torch.empty(0); // init

			private readonly int nc;
			private readonly int nl;
			private readonly int reg_max;
			private readonly int no;
			private readonly int[] stride;
			private readonly ModuleList<Sequential> cv2 = new ModuleList<Sequential>();
			private readonly ModuleList<Sequential> cv3 = new ModuleList<Sequential>();
			private readonly Module<Tensor, Tensor> dfl;

			public YolovDetect(int nc, int[] ch, bool legacy = false) : base("YolovDetect")
			{
				this.nc = nc; // number of classes
				this.nl = ch.Length;// number of detection layers
				this.reg_max = 16; // DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
				this.no = nc + this.reg_max * 4; // number of outputs per anchor
				this.stride = new int[] { 8, 16, 32 }; // strides computed during build

				int c2 = Math.Max(Math.Max(16, ch[0] / 4), this.reg_max * 4);
				int c3 = Math.Max(ch[0], Math.Min(this.nc, 100));// channels

				foreach (int x in ch)
				{
					cv2.append(Sequential(new Conv(x, c2, 3), new Conv(c2, c2, 3), nn.Conv2d(c2, 4 * this.reg_max, 1)));

					if (legacy)
					{
						cv3.append(Sequential(Sequential(new DWConv(x, x, 3), new Conv(x, c3, 1)), Sequential(new DWConv(c3, c3, 3), new Conv(c3, c3, 1)), nn.Conv2d(c3, this.nc, 1)));
					}
					else
					{
						cv3.append(Sequential(new Conv(x, c3, 3), new Conv(c3, c3, 3), nn.Conv2d(c3, this.nc, 1)));
					}
				}

				this.dfl = this.reg_max > 1 ? new DFL(this.reg_max) : nn.Identity();
				//RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				for (int i = 0; i < nl; i++)
				{
					x[i] = torch.cat([cv2[i].forward(x[i]), cv3[i].forward(x[i])], 1);
				}

				if (training)
				{
					return x;
				}
				else
				{
					Tensor y = _inference(x);
					return new Tensor[] { y }.Concat(x).ToArray();
				}
			}

			//Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.
			private Tensor _inference(Tensor[] x)
			{
				long[] shape = x[0].shape;  // BCHW

				List<Tensor> xi_mix = new List<Tensor>();
				foreach (var xi in x)
				{
					xi_mix.Add(xi.view(shape[0], this.no, -1));
				}
				Tensor x_cat = torch.cat(xi_mix, 2);

				if (this.shape != shape)
				{
					var (anchors, strides) = make_anchors(x, this.stride, 0.5f);
					this.anchors = anchors.transpose(0, 1);
					this.strides = strides.transpose(0, 1);
					this.shape = shape;
				}

				Tensor[] box_cls = x_cat.split([this.reg_max * 4, this.nc], 1);
				Tensor box = box_cls[0];
				Tensor cls = box_cls[1];
				Tensor dbox = decode_bboxes(this.dfl.forward(box), this.anchors.unsqueeze(0)) * this.strides;
				return torch.cat([dbox, cls.sigmoid()], 1);
			}

			// Decode bounding boxes.
			private Tensor decode_bboxes(Tensor bboxes, Tensor anchors)
			{
				return dist2bbox(bboxes, anchors, xywh: true, dim: 1);
			}

			// Transform distance(ltrb) to box(xywh or xyxy).
			private Tensor dist2bbox(Tensor distance, Tensor anchor_points, bool xywh = true, int dim = -1)
			{
				Tensor[] ltrb = distance.chunk(2, dim);
				Tensor lt = ltrb[0];
				Tensor rb = ltrb[1];

				Tensor x1y1 = anchor_points - lt;
				Tensor x2y2 = anchor_points + rb;

				if (xywh)
				{
					Tensor c_xy = (x1y1 + x2y2) / 2;
					Tensor wh = x2y2 - x1y1;
					return torch.cat([c_xy, wh], dim);  // xywh bbox
				}
				return torch.cat([x1y1, x2y2], dim); // xyxy bbox
			}

			private (Tensor, Tensor) make_anchors(Tensor[] feats, int[] strides, float grid_cell_offset = 0.5f)
			{
				torch.ScalarType dtype = feats[0].dtype;
				Device device = feats[0].device;
				List<Tensor> anchor_points = new List<Tensor>();
				List<Tensor> stride_tensor = new List<Tensor>();
				for (int i = 0; i < strides.Length; i++)
				{
					long h = feats[i].shape[2];
					long w = feats[i].shape[3];
					Tensor sx = torch.arange(w, device: device, dtype: dtype) + grid_cell_offset;  // shift x
					Tensor sy = torch.arange(h, device: device, dtype: dtype) + grid_cell_offset;  // shift y
					Tensor[] sy_sx = torch.meshgrid([sy, sx], indexing: "ij");
					sy = sy_sx[0];
					sx = sy_sx[1];
					anchor_points.Add(torch.stack([sx, sy], -1).view(-1, 2));
					stride_tensor.Add(torch.full([h * w, 1], strides[i], dtype: dtype, device: device));
				}
				return (torch.cat(anchor_points), torch.cat(stride_tensor));
			}

		}

		public class Proto : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Conv cv3;
			private readonly ConvTranspose2d upsample;
			public Proto(int c1, int c_ = 256, int c2 = 32) : base("Proto")
			{
				this.cv1 = new Conv(c1, c_, kernel_size: 3);
				this.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias: true);  // nn.Upsample(scale_factor=2, mode='nearest')
				this.cv2 = new Conv(c_, c_, kernel_size: 3);
				this.cv3 = new Conv(c_, c2, kernel_size: 1);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				return this.cv3.forward(this.cv2.forward(this.upsample.forward(this.cv1.forward(x)))).MoveToOuterDisposeScope();
			}
		}

		public class Segment : YolovDetect
		{
			private readonly int nm;
			private readonly int npr;
			private readonly Proto proto;
			private readonly int c4;
			private readonly ModuleList<Sequential> cv4 = new ModuleList<Sequential>();

			public Segment(int[] ch, int nc = 80, int nm = 32, int npr = 256, bool legacy = false) : base(nc, ch, legacy)
			{
				this.nm = nm; // number of masks
				this.npr = npr;  // number of protos
				this.proto = new Proto(ch[0], this.npr, this.nm);  // protos
				c4 = Math.Max(ch[0] / 4, this.nm);

				foreach (int x in ch)
				{
					cv4.append(Sequential(new Conv(x, c4, 3), new Conv(c4, c4, 3), nn.Conv2d(c4, this.nm, 1)));
				}
				//RegisterComponents();

			}

			public override Tensor[] forward(Tensor[] x)
			{
				Tensor p = this.proto.forward(x[0]); // mask protos
				long bs = p.shape[0]; //batch size

				var mc = torch.cat(this.cv4.Select((module, i) => module.forward(x[i]).view(bs, this.nm, -1)).ToArray(), dim: 2); // mask coefficients				x = base.forward(x);
				x = base.forward(x);
				if (this.training)
				{
					x = (x.Append(mc).Append(p)).ToArray();
					return x;
				}
				else
				{
					return [torch.cat([x[0], mc], dim: 1), x[1], x[2], x[3], p];
				}
			}
		}

	}
}

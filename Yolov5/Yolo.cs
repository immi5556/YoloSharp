using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Yolov5
{
	public class Yolo
	{
		private class Conv : Module<Tensor, Tensor>
		{
			internal readonly Conv2d conv;
			internal readonly BatchNorm2d bn;
			internal readonly bool act;

			public Conv(int in_channels, int out_channels, int kernel_size, int stride = 1, int? padding = null, int groups = 1, bool bias = true, bool act = true) : base("Conv")
			{
				if (padding == null)
				{
					padding = (kernel_size - 1) / 2;
				}

				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding.Value, groups: groups, bias: bias);
				bn = BatchNorm2d(out_channels);
				this.act = act;
				RegisterComponents();
			}
			public override Tensor forward(Tensor input)
			{
				var x = conv.forward(input);
				Module<Tensor,Tensor> ac = act ? SiLU() : Identity();
				return ac.forward(bn.forward(x));
			}
		}

		private class Bottleneck : Module<Tensor, Tensor>
		{
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			bool add;
			public Bottleneck(int inChannels, int outChannels, bool shortcut = true, int groups = 1, float e = 0.5f) : base("bottleneck")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, c, 1, 1);
				this.cv2 = new Conv(c, outChannels, 3, 1, groups: groups);
				this.add = shortcut && inChannels == outChannels;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return add ? input + cv2.forward(cv1.forward(input)) : cv2.forward(cv1.forward(input));
			}
		}

		private class C3 : Module<Tensor, Tensor>
		{
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			internal readonly Conv cv3;
			internal readonly Sequential m;

			public C3(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f) : base("C3")
			{
				int c = (int)(outChannels * e);
				this.cv1 = new Conv(inChannels, c, 1, 1);
				this.cv2 = new Conv(inChannels, c, 1, 1);
				this.cv3 = new Conv(2 * c, outChannels, 1);

				m = Sequential();
				for (int i = 0; i < n; i++)
				{
					m.append(new Bottleneck(c, c, shortcut, groups, e: 1.0f));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return cv3.forward(cat([m.forward(cv1.forward(input)), cv2.forward(input)], 1));
			}
		}

		private class SPPF : Module<Tensor, Tensor>
		{
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			internal readonly MaxPool2d m;

			public SPPF(int inChannels, int outChannels, int kernalSize = 5) : base("SPPF")
			{
				int c = inChannels / 2;
				cv1 = new Conv(inChannels, c, 1, 1);
				cv2 = new Conv(c * 4, outChannels, 1, 1);
				m = nn.MaxPool2d(kernelSize: kernalSize, stride: 1, padding: kernalSize / 2);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				var x = cv1.forward(input);
				var y1 = m.forward(x);
				var y2 = m.forward(y1);

				return cv2.forward(torch.cat(new[] { x, y1, y2, m.forward(y2) }, 1));
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
				this.no = nc + 5;// =85 每个类别需添加位置与置信度
				this.nl = anchors.Length;
				this.na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				float[] an = new float[anchors.Length * anchors[0].Length];
				for (int i = 0; i < anchors.Length; i++)
				{
					for (int j = 0; j < anchors[1].Length; j++)
					{
						an[i * anchors[0].Length + j] = anchors[i][j];
					}
				}
				this.anchors = torch.tensor(an, [anchors.Length, anchors[0].Length / 2, 2]);
				this.stride = torch.tensor(model_stride, ScalarType.Float32);
				grid = new List<Tensor>(nl);
				anchor_grid = new List<Tensor>(nl);
				for (int i = 0; i < nl; i++)
				{
					grid.Add(torch.empty(0));
					anchor_grid.Add(torch.empty(0));
				}
				for (int i = 0; i < ch.Length; i++)
				{
					m.append(Conv2d(ch[i], no * na, 1));
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
					x[i] = x[i].view(bs, this.na, this.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
					if (!this.training)
					{
						(this.grid[i], this.anchor_grid[i]) = _make_grid(nx, ny, i);
						Tensor[] re = x[i].sigmoid().split([2, 2, this.nc + 1], 4);
						Tensor xy = re[0];
						Tensor wh = re[1];
						Tensor conf = re[2];

						xy = (xy * 2 + this.grid[i]) * this.stride[i];  // xy
						wh = (wh * 2).pow(2) * this.anchor_grid[i];  // wh
						Tensor y = torch.cat([xy, wh, conf], 4);
						z.Add(y.view(bs, this.na * nx * ny, this.no));
					}
				}

				if (this.training)
				{
					return x;
				}
				else
				{
					var list = new List<Tensor>() { torch.cat(z, 1) };
					list.AddRange(x);
					return list.ToArray();
				}
			}

			private (Tensor, Tensor) _make_grid(int nx = 20, int ny = 20, int i = 0)
			{
				var d = this.anchors[i].device;
				var t = this.anchors[i].dtype;

				long[] shape = [1, this.na, ny, nx, 2]; // grid shape
				Tensor y = torch.arange(ny, t, d);
				Tensor x = torch.arange(nx, t, d);
				Tensor[] xy = torch.meshgrid([y, x], indexing: "ij");
				Tensor yv = xy[0];
				Tensor xv = xy[1];
				Tensor grid = torch.stack([xv, yv], 2).expand(shape) - 0.5f;  // add grid offset, i.e. y = 2.0 * x - 0.5

				Tensor anchor_grid = (this.anchors[i] * this.stride[i]).view([1, this.na, 1, 1, 2]).expand(shape);

				return (grid, anchor_grid);
			}
		}


		public class Yolov5 : Module<Tensor, Tensor[]>
		{
			internal readonly Sequential backbone;
			internal readonly ModuleList<Module<Tensor, Tensor>> head;
			internal readonly Detect dect;

			public Yolov5(int nc = 80, float depth_multiple = 0.33f, float width_multiple = 0.25f) : base("Yolov5")
			{
				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] ach = [[10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d], // P3/8
						[30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d],// P4/16
						[116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d]];   // P5/32

				int[] ch = [(int)(256 * width_multiple), (int)(512 * width_multiple), (int)(1024 * width_multiple)];

				backbone = Sequential(
					new Conv(3, (int)(64 * width_multiple), 6, 2, 2),       //P1

					new Conv((int)(64 * width_multiple), (int)(128 * width_multiple), 3, 2),        //P2
					new C3((int)(128 * width_multiple), (int)(128 * width_multiple), (int)(3 * depth_multiple)),

					new Conv((int)(128 * width_multiple), (int)(256 * width_multiple), 3, 2),       //P3
					new C3((int)(256 * width_multiple), (int)(256 * width_multiple), (int)(6 * depth_multiple)),

					new Conv((int)(256 * width_multiple), (int)(512 * width_multiple), 3, 2),       //P4
					new C3((int)(512 * width_multiple), (int)(512 * width_multiple), (int)(9 * depth_multiple)),

					new Conv((int)(512 * width_multiple), (int)(1024 * width_multiple), 3, 2),      //P5
					new C3((int)(1024 * width_multiple), (int)(1024 * width_multiple), (int)(3 * depth_multiple)),
					new SPPF((int)(1024 * width_multiple), (int)(1024 * width_multiple), 5)
				);


				head = new ModuleList<Module<Tensor, Tensor>>
				{
					new Conv((int)(1024 * width_multiple), (int)(512 * width_multiple), 1, 1),
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C3((int)(1024 * width_multiple), (int)(512 * width_multiple), (int)(3 * depth_multiple),  false),    // [2]

					new Conv((int)(512 * width_multiple), (int)(256 * width_multiple), 1, 1),
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C3((int)(512 * width_multiple), (int)(256 * width_multiple), (int)(3 * depth_multiple), false),      // [5]

					new Conv((int)(256 * width_multiple), (int)(256 * width_multiple), 3, 2),
					new C3((int)(512 * width_multiple), (int)(512 * width_multiple), (int)(3 * depth_multiple), false),      // [7]

					new Conv((int)(512 * width_multiple), (int)(512 * width_multiple), 3, 2),
					new C3((int)(1024 * width_multiple), (int)(1024 * width_multiple), (int)(3 * depth_multiple), false)     // [9]
				};

				dect = new Detect(nc, ch, ach);
				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 10; i++)
				{
					x = ((Module<Tensor, Tensor>)backbone[i]).forward(x);
					if (i == 0 || i == 2 || i == 4 || i == 6 || i == 9)
					{
						outputs.Add(x);
					}
				}

				// First block
				var p10 = head[0].forward(x);
				x = head[1].forward(p10);
				x = cat([x, outputs[3]], 1); // Concat with P4
				x = head[2].forward(x);

				// Second block
				var p14 = head[3].forward(x);
				x = head[4].forward(p14);
				x = cat([x, outputs[2]], 1); // Concat with P3
				var p3_out = head[5].forward(x);

				// Third block
				x = head[6].forward(p3_out);
				x = cat([x, p14], 1); // Concat with P4
				var p4_out = head[7].forward(x);

				// Fourth block
				x = head[8].forward(p4_out);
				x = cat([x, p10], 1); // Concat with P5
				var p5_out = head[9].forward(x);

				var list = dect.forward(new Tensor[] { p3_out, p4_out, p5_out });


				return list;
			}
		}


	}
}

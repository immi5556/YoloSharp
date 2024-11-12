using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Modules;

namespace YoloSharp
{
	internal class Yolo
	{
		public enum YoloSize
		{
			n = 0,
			s,
			m,
			l,
			x,
		}

		public class Yolov5 : Module<Tensor, Tensor[]>
		{
			private readonly Sequential backbone;
			private readonly ModuleList<Module<Tensor, Tensor>> head;
			private readonly Detect dect;

			public Yolov5(int nc = 80, YoloSize yoloSize = YoloSize.n) : base("Yolov5")
			{
				float depth_multiple = 0.34f;
				float width_multiple = 0.25f;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.25f;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.5f;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.67f;
							width_multiple = 0.75f;
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.34f;
							width_multiple = 1.25f;
							break;
						}
				}

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

				var list = dect.forward([p3_out, p4_out, p5_out]);
				return list;

			}

			public void Save(string path)
			{
				this.save(path);
			}
			public IEnumerable<Parameter> Parameters(bool recurse = true)
			{
				return this.parameters(recurse);
			}
		}

		public class Yolov8 : Module<Tensor, Tensor[]>
		{
			private readonly Sequential backbone;
			private readonly ModuleList<Module<Tensor, Tensor>> head;
			private readonly Detect dect;

			public Yolov8(int nc = 80, YoloSize yoloSize = YoloSize.n) : base("Yolov5")
			{
				float depth_multiple = 0.34f;
				float width_multiple = 0.25f;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.25f;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.5f;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.67f;
							width_multiple = 0.75f;
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.34f;
							width_multiple = 1.25f;
							break;
						}
				}

				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] ach = [[10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d], // P3/8
						[30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d],// P4/16
						[116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d]];   // P5/32

				int[] ch = [(int)(256 * width_multiple), (int)(512 * width_multiple), (int)(1024 * width_multiple)];

				backbone = Sequential(
					new Conv(3, (int)(64 * width_multiple), 3, 2),       //P1

					new Conv((int)(64 * width_multiple), (int)(128 * width_multiple), 3, 2),        //P2
					new C2f((int)(128 * width_multiple), (int)(128 * width_multiple), (int)(3 * depth_multiple), true),

					new Conv((int)(128 * width_multiple), (int)(256 * width_multiple), 3, 2),       //P3
					new C2f((int)(256 * width_multiple), (int)(256 * width_multiple), (int)(6 * depth_multiple), true),

					new Conv((int)(256 * width_multiple), (int)(512 * width_multiple), 3, 2),       //P4
					new C2f((int)(512 * width_multiple), (int)(512 * width_multiple), (int)(6 * depth_multiple), true),

					new Conv((int)(512 * width_multiple), (int)(1024 * width_multiple), 3, 2),      //P5
					new C2f((int)(1024 * width_multiple), (int)(1024 * width_multiple), (int)(6 * depth_multiple), true),
					new SPPF((int)(1024 * width_multiple), (int)(1024 * width_multiple), 5)
				);


				head = new ModuleList<Module<Tensor, Tensor>>
				{
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C2f((int)(1536 * width_multiple), (int)(512 * width_multiple), (int)(3 * depth_multiple)),    // [2]

					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C2f((int)(768 * width_multiple), (int)(256 * width_multiple), (int)(3 * depth_multiple)),      // [5]

					new Conv((int)(256 * width_multiple), (int)(256 * width_multiple), 3, 2),
					new C2f((int)(768 * width_multiple), (int)(512 * width_multiple), (int)(3 * depth_multiple)),      // [7]

					new Conv((int)(512 * width_multiple), (int)(512 * width_multiple), 3, 2),
					new C2f((int)(1536 * width_multiple), (int)(1024 * width_multiple), (int)(3 * depth_multiple))     // [9]
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
				x = head[0].forward(x);
				x = cat([x, outputs[3]], 1); // Concat with P4
				var p12 = head[1].forward(x);

				// Second block
				x = head[2].forward(p12);
				x = cat([x, outputs[2]], 1); // Concat with P3
				var p3_out = head[3].forward(x);

				// Third block
				x = head[4].forward(p3_out);
				x = cat([x, p12], 1); // Concat with P4
				var p4_out = head[5].forward(x);

				// Fourth block
				x = head[6].forward(p4_out);
				x = cat([x, outputs[4]], 1); // Concat with P5
				var p5_out = head[7].forward(x);

				var list = dect.forward([p3_out, p4_out, p5_out]);
				return list;

			}

			public void Save(string path)
			{
				this.save(path);
			}
			public IEnumerable<Parameter> Parameters(bool recurse = true)
			{
				return this.parameters(recurse);
			}
		}

		public class Yolov11 : Module<Tensor, Tensor[]>
		{
			private readonly Sequential backbone;
			private readonly ModuleList<Module<Tensor, Tensor>> head;
			private readonly Detect dect;

			public Yolov11(int nc = 80, YoloSize yoloSize = YoloSize.n) : base("Yolov11")
			{
				float depth_multiple = 0.5f;
				float width_multiple = 0.25f;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.5f;
							width_multiple = 0.25f;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.5f;
							width_multiple = 0.5f;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.5f;
							width_multiple = 1.0f;
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.5f;
							break;
						}
				}

				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] ach = [[10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d], // P3/8
						[30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d],// P4/16
						[116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d]];   // P5/32

				int[] ch = [(int)(256 * width_multiple), (int)(512 * width_multiple), (int)(1024 * width_multiple)];


				backbone = Sequential(
					new Conv(3, (int)(64 * width_multiple), 3, 2),       // 0-P1/2

					new Conv((int)(64 * width_multiple), (int)(128 * width_multiple), 3, 2),        // 0-P2/4
					new C3k2((int)(128 * width_multiple), (int)(256 * width_multiple), (int)(2 * depth_multiple), shortcut: true, e: 0.25f),

					new Conv((int)(256 * width_multiple), (int)(256 * width_multiple), 3, 2),       //P3
					new C3k2((int)(256 * width_multiple), (int)(512 * width_multiple), (int)(2 * depth_multiple), shortcut: true, e: 0.25f),

					new Conv((int)(512 * width_multiple), (int)(512 * width_multiple), 3, 2),       //P4
					new C3k2((int)(512 * width_multiple), (int)(512 * width_multiple), (int)(2 * depth_multiple), c3k: true, shortcut: true),

					new Conv((int)(512 * width_multiple), (int)(1024 * width_multiple), 3, 2),      //P5
					new C3k2((int)(1024 * width_multiple), (int)(1024 * width_multiple), (int)(2 * depth_multiple), c3k: true, shortcut: true),
					new SPPF((int)(1024 * width_multiple), (int)(1024 * width_multiple), 5),
					new C2PSA((int)(1024 * width_multiple), (int)(1024 * width_multiple))
				);

				head = new ModuleList<Module<Tensor, Tensor>>
				{
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C3k2((int)(1536 * width_multiple), (int)(512 * width_multiple), (int)(2 * depth_multiple),  shortcut: true),

					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new C3k2((int)(1024 * width_multiple), (int)(256 * width_multiple), (int)(2 * depth_multiple),  shortcut: true),

					new Conv((int)(256 * width_multiple), (int)(256 * width_multiple), 3, 2),
					new C3k2((int)(768 * width_multiple), (int)(512 * width_multiple), (int)(2 * depth_multiple),  shortcut: true),

					new Conv((int)(512 * width_multiple), (int)(512 * width_multiple), 3, 2),
					new C3k2((int)(1536 * width_multiple), (int)(1024 * width_multiple), (int)(2 * depth_multiple), c3k:true, shortcut: true),
				};

				dect = new Detect(nc, ch, ach);

				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < backbone.Count; i++)
				{
					var md = ((Module<Tensor, Tensor>)backbone[i]);
					x = md.forward(x);
					if (i == 4 || i == 6 || i == 10)
					{
						outputs.Add(x);
					}
				}

				x = head[0].forward(x);
				x = cat([x, outputs[1]], dim: 1);
				Tensor p13 = head[1].forward(x);

				x = head[2].forward(p13);
				x = cat([x, outputs[0]], dim: 1);
				Tensor p16 = head[3].forward(x);

				x = head[4].forward(p16);
				x = cat([x, p13], dim: 1);
				Tensor p19 = head[5].forward(x);

				x = head[6].forward(p19);
				x = cat([x, outputs[2]], dim: 1);
				Tensor p22 = head[7].forward(x);

				var list = dect.forward([p16, p19, p22]);
				return list;
			}

			public void Save(string path)
			{
				this.save(path);
			}
			public IEnumerable<Parameter> Parameters(bool recurse = true)
			{
				return this.parameters(recurse);
			}


		}

	}
}

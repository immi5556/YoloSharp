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
			private readonly ModuleList<Module> model;

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

				int widthSize64 = (int)(64 * width_multiple);
				int widthSize128 = (int)(128 * width_multiple);
				int widthSize256 = (int)(256 * width_multiple);
				int widthSize512 = (int)(512 * width_multiple);
				int widthSize1024 = (int)(1024 * width_multiple);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);


				int[] ch = [widthSize256, widthSize512, widthSize1024];

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 6, 2, 2),                                           // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2),                    // 1-P2/4
					new C3(widthSize128, widthSize128, depthSize3),
					new Conv(widthSize128, widthSize256, 3, 2),                   // 3-P3/8
					new C3(widthSize256, widthSize256, depthSize6),
					new Conv(widthSize256, widthSize512, 3, 2),                   // 5-P4/16
					new C3(widthSize512, widthSize512, depthSize9),
					new Conv(widthSize512, widthSize1024, 3, 2),                  // 7-P5/32
					new C3(widthSize1024, widthSize1024, depthSize3),
					new SPPF(widthSize1024, widthSize1024, 5),

					// head:
					new Conv(widthSize1024, widthSize512, 1, 1),
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widthSize1024, widthSize512, depthSize3, false),    // 13

					new Conv(widthSize512, widthSize256, 1, 1),
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widthSize512, widthSize256, depthSize3, false),      // 17 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2),
					new Concat(),                                                                               // cat head P4
					new C3(widthSize512, widthSize512, depthSize3, false),      // 20 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2),
					new Concat(),                                                                               // cat head P5
					new C3(widthSize1024, widthSize1024, depthSize3, false),     // 23 (P5/32-large)

					new Yolov5Detect(nc, ch, ach)                                                               // Detect(P3, P4, P5)
				);
				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 10; i++)
				{
					x = ((Module<Tensor, Tensor>)model[i]).forward(x);
					if (i == 4 || i == 6)
					{
						outputs.Add(x);
					}
				}

				// First block
				var p10 = ((Module<Tensor, Tensor>)model[10]).forward(x);
				x = ((Module<Tensor, Tensor>)model[11]).call(p10);
				x = ((Module<Tensor[], Tensor>)model[12]).call([x, outputs[1]]); // Concat with P4
				x = ((Module<Tensor, Tensor>)model[13]).call(x);

				// Second block
				var p14 = ((Module<Tensor, Tensor>)model[14]).call(x);
				x = ((Module<Tensor, Tensor>)model[15]).call(p14);
				x = ((Module<Tensor[], Tensor>)model[16]).call([x, outputs[0]]); // Concat with P3
				var p3_out = ((Module<Tensor, Tensor>)model[17]).call(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[18]).call(p3_out);
				x = ((Module<Tensor[], Tensor>)model[19]).call([x, p14]); // Concat with P4
				var p4_out = ((Module<Tensor, Tensor>)model[20]).call(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[21]).call(p4_out);
				x = ((Module<Tensor[], Tensor>)model[22]).call([x, p10]); // Concat with P5
				var p5_out = ((Module<Tensor, Tensor>)model[23]).call(x);

				var list = ((Module<Tensor[], Tensor[]>)model[24]).forward([p3_out, p4_out, p5_out]);
				return list;

			}

		}

		public class Yolov8 : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov8(int nc = 80, YoloSize yoloSize = YoloSize.n) : base("Yolov5")
			{
				float depth_multiple = 0.34f;
				float width_multiple = 0.25f;
				int maxSize = 1024;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.25f;
							maxSize = 1024;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.5f;
							maxSize = 1024;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.67f;
							width_multiple = 0.75f;
							maxSize = 576;          // max size =  768 in yolo8.yaml, but it can be 576 in real yolo model and yolov8m.pt
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							maxSize = 512;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.25f;
							maxSize = 640;      // max size =  512 in yolo8.yaml, but it can be 640 in real yolo model and yolov8x.pt
							break;
						}
				}

				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] ach = [[10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d], // P3/8
						[30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d],// P4/16
						[116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d]];   // P5/32

				int widthSize64 = Math.Min((int)(64 * width_multiple), maxSize);
				int widthSize128 = Math.Min((int)(128 * width_multiple), maxSize);
				int widthSize256 = Math.Min((int)(256 * width_multiple), maxSize);
				int widthSize512 = Math.Min((int)(512 * width_multiple), maxSize);
				int widthSize1024 = Math.Min((int)(1024 * width_multiple), maxSize);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);

				int[] ch = [widthSize256, widthSize512, widthSize1024];

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 3, 2),                                                                     // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2),                                                          // 1-P2/4
					new C2f(widthSize128, widthSize128, depthSize3, true),
					new Conv(widthSize128, widthSize256, 3, 2),                                                         // 3-P3/8
					new C2f(widthSize256, widthSize256, depthSize6, true),
					new Conv(widthSize256, widthSize512, 3, 2),                                                         // 5-P4/16
					new C2f(widthSize512, widthSize512, depthSize6, true),
					new Conv(widthSize512, widthSize1024, 3, 2),                                                        // 7-P5/32
					new C2f(widthSize1024, widthSize1024, depthSize3, true),
					new SPPF(widthSize1024, widthSize1024, 5),                                                      // 9

					// head:
					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new C2f(widthSize512 + widthSize1024, widthSize512, depthSize3),                                    // 12

					Upsample(scale_factor: [2, 2], mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new C2f(widthSize256 + widthSize512, widthSize256, depthSize3),                                     // 15 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2),
					new Concat(),                                                                                       // cat head P4
					new C2f(widthSize256 + widthSize512, widthSize512, depthSize3),                                    // 18 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2),
					new Concat(),                                                                                       // cat head P5
					new C2f(widthSize1024 + widthSize512, widthSize1024, depthSize3),                                  // 21 (P5/32-large)

					new Yolov8Detect(nc, ch)                                                                            // Detect(P3, P4, P5)
					);
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 10; i++)
				{
					x = ((Module<Tensor, Tensor>)model[i]).forward(x);
					if (i == 4 || i == 6 || i == 9)
					{
						outputs.Add(x);
					}
				}

				// First block
				x = ((Module<Tensor, Tensor>)model[10]).forward(x);
				x = ((Module<Tensor[], Tensor>)model[11]).forward([x, outputs[1]]);
				var p12 = ((Module<Tensor, Tensor>)model[12]).forward(x);

				// Second block
				x = ((Module<Tensor, Tensor>)model[13]).forward(p12);
				x = ((Module<Tensor[], Tensor>)model[14]).forward([x, outputs[0]]); // Concat with P3
				var p15 = ((Module<Tensor, Tensor>)model[15]).forward(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[16]).forward(p15);
				x = ((Module<Tensor[], Tensor>)model[17]).forward([x, p12]); // Concat with P4
				var p18 = ((Module<Tensor, Tensor>)model[18]).forward(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[19]).forward(p18);
				x = ((Module<Tensor[], Tensor>)model[20]).forward([x, outputs[2]]); // Concat with P5
				var p21 = ((Module<Tensor, Tensor>)model[21]).forward(x);

				var list = ((Module<Tensor[], Tensor[]>)model[22]).forward([p15, p18, p21]);
				return list;
			}

		}

		public class Yolov11 : Module<Tensor, Tensor[]>
		{
			private readonly Sequential backbone;
			private readonly ModuleList<Module<Tensor, Tensor>> head;
			private readonly Yolov5Detect dect;
			private readonly Yolov8Detect yolov8Detect;
			private int[] strides;

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
				strides = ch;


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

				dect = new Yolov5Detect(nc, ch, ach);
				yolov8Detect = new Yolov8Detect(nc, ch);
				yolov8Detect.end2end = true;

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

				var lll = yolov8Detect.forward([p16, p19, p22]);

				Loss.Yolov8DetectionLoss v8DetectionLoss = new Loss.Yolov8DetectionLoss();

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

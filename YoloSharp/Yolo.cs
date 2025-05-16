using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Modules;

namespace YoloSharp
{
	internal class Yolo
	{
		public class Yolov5 : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov5(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base("Yolov5")
			{
				(float depth_multiple, float width_multiple) = yoloSize switch
				{
					YoloSize.n => (0.34f, 0.25f),
					YoloSize.s => (0.34f, 0.5f),
					YoloSize.m => (0.67f, 0.75f),
					YoloSize.l => (1.0f, 1.0f),
					YoloSize.x => (1.34f, 1.25f),
					_ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
				};

				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;

				float[][] ach = new float[][] { new float[] { 10 / p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23 / p3_d }, // P3/8
						new float[] { 30 / p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119 / p4_d },// P4/16
						new float[] { 116 / p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326 / p5_d }};   // P5/32

				int widthSize64 = (int)(64 * width_multiple);
				int widthSize128 = (int)(128 * width_multiple);
				int widthSize256 = (int)(256 * width_multiple);
				int widthSize512 = (int)(512 * width_multiple);
				int widthSize1024 = (int)(1024 * width_multiple);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);


				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widthSize128, widthSize128, depthSize3, device: device, dtype: dtype),
					new Conv(widthSize128, widthSize256, 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widthSize256, widthSize256, depthSize6, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize512, 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widthSize512, widthSize512, depthSize9, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widthSize1024, widthSize1024, depthSize3, device: device, dtype: dtype),
					new SPPF(widthSize1024, widthSize1024, 5, device: device, dtype: dtype),

					// head:
					new Conv(widthSize1024, widthSize512, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widthSize1024, widthSize512, depthSize3, false, device: device, dtype: dtype),    // 13

					new Conv(widthSize512, widthSize256, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widthSize512, widthSize256, depthSize3, false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widthSize512, widthSize512, depthSize3, false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widthSize1024, widthSize1024, depthSize3, false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new Yolov5Detect(nc, ch, ach, device: device, dtype: dtype)                                                               // Detect(P3, P4, P5)
				);
				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
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
				x = ((Module<Tensor[], Tensor>)model[12]).call(new Tensor[] { x, outputs[1] }); // Concat with P4
				x = ((Module<Tensor, Tensor>)model[13]).call(x);

				// Second block
				var p14 = ((Module<Tensor, Tensor>)model[14]).call(x);
				x = ((Module<Tensor, Tensor>)model[15]).call(p14);
				x = ((Module<Tensor[], Tensor>)model[16]).call(new Tensor[] { x, outputs[0] }); // Concat with P3
				var p3_out = ((Module<Tensor, Tensor>)model[17]).call(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[18]).call(p3_out);
				x = ((Module<Tensor[], Tensor>)model[19]).call(new Tensor[] { x, p14 }); // Concat with P4
				var p4_out = ((Module<Tensor, Tensor>)model[20]).call(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[21]).call(p4_out);
				x = ((Module<Tensor[], Tensor>)model[22]).call(new Tensor[] { x, p10 }); // Concat with P5
				var p5_out = ((Module<Tensor, Tensor>)model[23]).call(x);

				var list = ((Module<Tensor[], Tensor[]>)model[24]).forward(new Tensor[] { p3_out, p4_out, p5_out });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;

			}

		}

		public class Yolov5u : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov5u(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov5u))
			{
				(float depth_multiple, float width_multiple) = yoloSize switch
				{
					YoloSize.n => (0.34f, 0.25f),
					YoloSize.s => (0.34f, 0.5f),
					YoloSize.m => (0.67f, 0.75f),
					YoloSize.l => (1.0f, 1.0f),
					YoloSize.x => (1.34f, 1.25f),
					_ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
				};

				int widthSize64 = (int)(64 * width_multiple);
				int widthSize128 = (int)(128 * width_multiple);
				int widthSize256 = (int)(256 * width_multiple);
				int widthSize512 = (int)(512 * width_multiple);
				int widthSize1024 = (int)(1024 * width_multiple);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);

				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widthSize128, widthSize128, depthSize3, device: device, dtype: dtype),
					new Conv(widthSize128, widthSize256, 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widthSize256, widthSize256, depthSize6, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize512, 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widthSize512, widthSize512, depthSize9, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widthSize1024, widthSize1024, depthSize3, device: device, dtype: dtype),
					new SPPF(widthSize1024, widthSize1024, 5, device: device, dtype: dtype),

					// head:
					new Conv(widthSize1024, widthSize512, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widthSize1024, widthSize512, depthSize3, false, device: device, dtype: dtype),    // 13

					new Conv(widthSize512, widthSize256, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widthSize512, widthSize256, depthSize3, false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widthSize512, widthSize512, depthSize3, false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widthSize1024, widthSize1024, depthSize3, false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new YolovDetect(nc, ch, device: device, dtype: dtype)                                                               // Detect(P3, P4, P5)
				);
				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
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
				x = ((Module<Tensor[], Tensor>)model[12]).call(new Tensor[] { x, outputs[1] }); // Concat with P4
				x = ((Module<Tensor, Tensor>)model[13]).call(x);

				// Second block
				var p14 = ((Module<Tensor, Tensor>)model[14]).call(x);
				x = ((Module<Tensor, Tensor>)model[15]).call(p14);
				x = ((Module<Tensor[], Tensor>)model[16]).call(new Tensor[] { x, outputs[0] }); // Concat with P3
				var p3_out = ((Module<Tensor, Tensor>)model[17]).call(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[18]).call(p3_out);
				x = ((Module<Tensor[], Tensor>)model[19]).call(new Tensor[] { x, p14 }); // Concat with P4
				var p4_out = ((Module<Tensor, Tensor>)model[20]).call(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[21]).call(p4_out);
				x = ((Module<Tensor[], Tensor>)model[22]).call(new Tensor[] { x, p10 }); // Concat with P5
				var p5_out = ((Module<Tensor, Tensor>)model[23]).call(x);

				var list = ((Module<Tensor[], Tensor[]>)model[24]).forward(new Tensor[] { p3_out, p4_out, p5_out });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;

			}

		}

		public class Yolov8 : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov8(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov8))
			{
				var (depth_multiple, width_multiple, max_channels) = yoloSize switch
				{
					YoloSize.n => (0.34f, 0.25f, 1024),
					YoloSize.s => (0.34f, 0.5f, 1024),
					YoloSize.m => (0.67f, 0.75f, 576),
					YoloSize.l => (1.0f, 1.0f, 512),
					YoloSize.x => (1.0f, 1.25f, 640),
					_ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
				};

				int[] widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => Math.Min((int)(w * width_multiple), max_channels)).ToArray();
				int[] depths = new List<int> { 3, 6, 9 }.Select(d => (int)(d * depth_multiple)).ToArray();
				int[] ch = { widths[2], widths[3], widths[4] };

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C2f(widths[1], widths[1], depths[0], true, device: device, dtype: dtype),
					new Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),
					new C2f(widths[2], widths[2], depths[1], true, device: device, dtype: dtype),
					new Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),
					new C2f(widths[3], widths[3], depths[1], true, device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C2f(widths[4], widths[4], depths[0], true, device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[3] + widths[4], widths[3], depths[0], device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[2], depths[0], device: device, dtype: dtype),

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[3], depths[0], device: device, dtype: dtype),

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[4] + widths[3], widths[4], depths[0], device: device, dtype: dtype),

					new YolovDetect(nc, ch, device: device, dtype: dtype)
				);

				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
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
				x = ((Module<Tensor[], Tensor>)model[11]).forward(new Tensor[] { x, outputs[1] });
				var p12 = ((Module<Tensor, Tensor>)model[12]).forward(x);

				// Second block
				x = ((Module<Tensor, Tensor>)model[13]).forward(p12);
				x = ((Module<Tensor[], Tensor>)model[14]).forward(new Tensor[] { x, outputs[0] }); // Concat with P3
				var p15 = ((Module<Tensor, Tensor>)model[15]).forward(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[16]).forward(p15);
				x = ((Module<Tensor[], Tensor>)model[17]).forward(new Tensor[] { x, p12 }); // Concat with P4
				var p18 = ((Module<Tensor, Tensor>)model[18]).forward(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[19]).forward(p18);
				x = ((Module<Tensor[], Tensor>)model[20]).forward(new Tensor[] { x, outputs[2] }); // Concat with P5
				var p21 = ((Module<Tensor, Tensor>)model[21]).forward(x);

				var list = ((Module<Tensor[], Tensor[]>)model[22]).forward(new Tensor[] { p15, p18, p21 });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;
			}

		}

		public class Yolov11 : Module<Tensor, Tensor[]>
		{
			ModuleList<Module> model;

			public Yolov11(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov11))
			{
				(float depth_multiple, float width_multiple, int max_channels, bool useC3k) = yoloSize switch
				{
					YoloSize.n => (0.5f, 0.25f, 1024, false),
					YoloSize.s => (0.5f, 0.5f, 1024, false),
					YoloSize.m => (0.5f, 1.0f, 512, true),
					YoloSize.l => (1.0f, 1.0f, 512, true),
					YoloSize.x => (1.0f, 1.5f, 768, true),
					_ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
				};

				int[] baseWidths = { 64, 128, 256, 512, 1024 };
				int[] widths = baseWidths.Select(w => Math.Min((int)(w * width_multiple), max_channels)).ToArray();
				int depthSize = (int)(2 * depth_multiple);
				int[] ch = { widths[2], widths[3], widths[4] };

				model = new ModuleList<Module>(
					new Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[3], widths[3], depthSize, c3k: true, device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[4], widths[4], depthSize, c3k: true, device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),
					new C2PSA(widths[4], widths[4], depthSize, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[3] + widths[3], widths[2], depthSize, useC3k, device: device, dtype: dtype),

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[3] + widths[2], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),

					new YolovDetect(nc, ch, false, device: device, dtype: dtype)
				);

				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 11; i++)
				{
					Module<Tensor, Tensor> md = ((Module<Tensor, Tensor>)model[i]);
					x = md.forward(x);
					if (i == 4 || i == 6 || i == 10)
					{
						outputs.Add(x);
					}
				}

				x = ((Module<Tensor, Tensor>)model[11]).forward(x);
				x = ((Module<Tensor[], Tensor>)model[12]).forward(new Tensor[] { x, outputs[1] });
				Tensor p13 = ((Module<Tensor, Tensor>)model[13]).forward(x);

				x = ((Module<Tensor, Tensor>)model[14]).forward(p13);
				x = ((Module<Tensor[], Tensor>)model[15]).forward(new Tensor[] { x, outputs[0] });
				Tensor p16 = ((Module<Tensor, Tensor>)model[16]).forward(x);

				x = ((Module<Tensor, Tensor>)model[17]).forward(p16);
				x = ((Module<Tensor[], Tensor>)model[18]).forward(new Tensor[] { x, p13 });
				Tensor p19 = ((Module<Tensor, Tensor>)model[19]).forward(x);

				x = ((Module<Tensor, Tensor>)model[20]).forward(p19);
				x = ((Module<Tensor[], Tensor>)model[21]).forward(new Tensor[] { x, outputs[2] });
				Tensor p22 = ((Module<Tensor, Tensor>)model[22]).forward(x);

				Tensor[] list = ((Module<Tensor[], Tensor[]>)model[23]).forward(new Tensor[] { p16, p19, p22 });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;
			}
		}

		public class Yolov12 : Module<Tensor, Tensor[]>
		{
			ModuleList<Module> model;

			public Yolov12(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov12))
			{
				(float depth_multiple, float width_multiple, int max_channels, bool useC3k, int n_nultiple, bool useResidual, float mlp_ratio) = yoloSize switch
				{
					YoloSize.n => (0.5f, 0.25f, 1024, false, 1, false, 2.0f),
					YoloSize.s => (0.5f, 0.5f, 1024, false, 1, false, 2.0f),
					YoloSize.m => (0.5f, 1.0f, 512, true, 1, false, 2.0f),
					YoloSize.l => (1.0f, 1.0f, 512, true, 2, true, 1.2f),
					YoloSize.x => (1.0f, 1.5f, 768, true, 2, true, 1.2f),
					_ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
				};

				int widthSize64 = Math.Min((int)(64 * width_multiple), max_channels);
				int widthSize128 = Math.Min((int)(128 * width_multiple), max_channels);
				int widthSize256 = Math.Min((int)(256 * width_multiple), max_channels);
				int widthSize512 = Math.Min((int)(512 * width_multiple), max_channels);
				int widthSize1024 = Math.Min((int)(1024 * width_multiple), max_channels);
				int depthSize2 = (int)(2 * depth_multiple);

				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };

				model = new ModuleList<Module>(
					new Conv(3, widthSize64, 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
					new C3k2(widthSize128, widthSize256, depthSize2, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
					new C3k2(widthSize256, widthSize512, depthSize2, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
					new A2C2f(widthSize512, widthSize512, n: 2 * n_nultiple, a2: true, area: 4, useResidual, mlp_ratio, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
					new A2C2f(widthSize1024, widthSize1024, n: 2 * n_nultiple, a2: true, area: 1, useResidual, mlp_ratio, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new A2C2f(widthSize1024 + widthSize512, widthSize512, n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                   // 11

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new A2C2f(widthSize512 + widthSize512, widthSize256, n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                    // 14 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P4
					new A2C2f(widthSize512 + widthSize256, widthSize512, n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                        // 17 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P5
					new C3k2(widthSize1024 + widthSize512, widthSize1024, depthSize2, c3k: true, device: device, dtype: dtype),                       // 20 (P5/32-large)

					new YolovDetect(nc, ch, false, device: device, dtype: dtype)                                                                       // Detect(P3, P4, P5)
					);
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 9; i++)
				{
					var md = ((Module<Tensor, Tensor>)model[i]);
					x = md.forward(x);
					if (i == 4 || i == 6 || i == 8)
					{
						outputs.Add(x);
					}
				}

				x = ((Module<Tensor, Tensor>)model[9]).forward(x);
				x = ((Module<Tensor[], Tensor>)model[10]).forward(new Tensor[] { x, outputs[1] });
				Tensor p11 = ((Module<Tensor, Tensor>)model[11]).forward(x);

				x = ((Module<Tensor, Tensor>)model[12]).forward(p11);
				x = ((Module<Tensor[], Tensor>)model[13]).forward(new Tensor[] { x, outputs[0] });
				Tensor p14 = ((Module<Tensor, Tensor>)model[14]).forward(x);

				x = ((Module<Tensor, Tensor>)model[15]).forward(p14);
				x = ((Module<Tensor[], Tensor>)model[16]).forward(new Tensor[] { x, p11 });
				Tensor p17 = ((Module<Tensor, Tensor>)model[17]).forward(x);

				x = ((Module<Tensor, Tensor>)model[18]).forward(p17);
				x = ((Module<Tensor[], Tensor>)model[19]).forward(new Tensor[] { x, outputs[2] });
				Tensor p20 = ((Module<Tensor, Tensor>)model[20]).forward(x);

				var list = ((Module<Tensor[], Tensor[]>)model[21]).forward(new Tensor[] { p14, p17, p20 });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;
			}
		}

		public class Yolov5uSegment : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov5uSegment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov5uSegment))
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

				int widthSize64 = (int)(64 * width_multiple);
				int widthSize128 = (int)(128 * width_multiple);
				int widthSize256 = (int)(256 * width_multiple);
				int widthSize512 = (int)(512 * width_multiple);
				int widthSize1024 = (int)(1024 * width_multiple);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);

				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widthSize128, widthSize128, depthSize3, device: device, dtype: dtype),
					new Conv(widthSize128, widthSize256, 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widthSize256, widthSize256, depthSize6, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize512, 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widthSize512, widthSize512, depthSize9, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widthSize1024, widthSize1024, depthSize3, device: device, dtype: dtype),
					new SPPF(widthSize1024, widthSize1024, 5, device: device, dtype: dtype),

					// head:
					new Conv(widthSize1024, widthSize512, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widthSize1024, widthSize512, depthSize3, false, device: device, dtype: dtype),    // 13

					new Conv(widthSize512, widthSize256, 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widthSize512, widthSize256, depthSize3, false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widthSize512, widthSize512, depthSize3, false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widthSize1024, widthSize1024, depthSize3, false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new Segment(ch, nc, npr: widthSize256, device: device, dtype: dtype)                                                      // Detect(P3, P4, P5)
				);
				RegisterComponents();

			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
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
				x = ((Module<Tensor[], Tensor>)model[12]).call(new Tensor[] { x, outputs[1] }); // Concat with P4
				x = ((Module<Tensor, Tensor>)model[13]).call(x);

				// Second block
				var p14 = ((Module<Tensor, Tensor>)model[14]).call(x);
				x = ((Module<Tensor, Tensor>)model[15]).call(p14);
				x = ((Module<Tensor[], Tensor>)model[16]).call(new Tensor[] { x, outputs[0] }); // Concat with P3
				var p3_out = ((Module<Tensor, Tensor>)model[17]).call(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[18]).call(p3_out);
				x = ((Module<Tensor[], Tensor>)model[19]).call(new Tensor[] { x, p14 }); // Concat with P4
				var p4_out = ((Module<Tensor, Tensor>)model[20]).call(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[21]).call(p4_out);
				x = ((Module<Tensor[], Tensor>)model[22]).call(new Tensor[] { x, p10 }); // Concat with P5
				var p5_out = ((Module<Tensor, Tensor>)model[23]).call(x);

				var list = ((Module<Tensor[], Tensor[]>)model[24]).forward(new Tensor[] { p3_out, p4_out, p5_out });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;

			}

		}

		public class Yolov8Segment : Module<Tensor, Tensor[]>
		{
			private readonly ModuleList<Module> model;

			public Yolov8Segment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base("Yolov8Segment")
			{
				float depth_multiple = 0.34f;
				float width_multiple = 0.25f;
				int max_channels = 1024;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.25f;
							max_channels = 1024;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.34f;
							width_multiple = 0.5f;
							max_channels = 1024;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.67f;
							width_multiple = 0.75f;
							max_channels = 576;          // max size =  768 in yolo8.yaml, but it can be 576 in real yolo model and yolov8m.pt
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							max_channels = 512;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.25f;
							max_channels = 640;      // max size =  512 in yolo8.yaml, but it can be 640 in real yolo model and yolov8x.pt
							break;
						}
				}

				int widthSize64 = Math.Min((int)(64 * width_multiple), max_channels);
				int widthSize128 = Math.Min((int)(128 * width_multiple), max_channels);
				int widthSize256 = Math.Min((int)(256 * width_multiple), max_channels);
				int widthSize512 = Math.Min((int)(512 * width_multiple), max_channels);
				int widthSize1024 = Math.Min((int)(1024 * width_multiple), max_channels);
				int depthSize3 = (int)(3 * depth_multiple);
				int depthSize6 = (int)(6 * depth_multiple);
				int depthSize9 = (int)(9 * depth_multiple);

				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };

				model = ModuleList<Module>(
					// backbone:
					new Conv(3, widthSize64, 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
					new C2f(widthSize128, widthSize128, depthSize3, true, device: device, dtype: dtype),
					new Conv(widthSize128, widthSize256, 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
					new C2f(widthSize256, widthSize256, depthSize6, true, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize512, 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
					new C2f(widthSize512, widthSize512, depthSize6, true, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
					new C2f(widthSize1024, widthSize1024, depthSize3, true, device: device, dtype: dtype),
					new SPPF(widthSize1024, widthSize1024, 5, device: device, dtype: dtype),                                                          // 9

					// head:
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new C2f(widthSize512 + widthSize1024, widthSize512, depthSize3, device: device, dtype: dtype),                                    // 12

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new C2f(widthSize256 + widthSize512, widthSize256, depthSize3, device: device, dtype: dtype),                                     // 15 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P4
					new C2f(widthSize256 + widthSize512, widthSize512, depthSize3, device: device, dtype: dtype),                                    // 18 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P5
					new C2f(widthSize1024 + widthSize512, widthSize1024, depthSize3, device: device, dtype: dtype),                                  // 21 (P5/32-large)

					new Segment(ch, nc, npr: widthSize256, device: device, dtype: dtype)                                                             // Detect(P3, P4, P5)
					);
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
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
				x = ((Module<Tensor[], Tensor>)model[11]).forward(new Tensor[] { x, outputs[1] });
				var p12 = ((Module<Tensor, Tensor>)model[12]).forward(x);

				// Second block
				x = ((Module<Tensor, Tensor>)model[13]).forward(p12);
				x = ((Module<Tensor[], Tensor>)model[14]).forward(new Tensor[] { x, outputs[0] }); // Concat with P3
				var p15 = ((Module<Tensor, Tensor>)model[15]).forward(x);

				// Third block
				x = ((Module<Tensor, Tensor>)model[16]).forward(p15);
				x = ((Module<Tensor[], Tensor>)model[17]).forward(new Tensor[] { x, p12 }); // Concat with P4
				var p18 = ((Module<Tensor, Tensor>)model[18]).forward(x);

				// Fourth block
				x = ((Module<Tensor, Tensor>)model[19]).forward(p18);
				x = ((Module<Tensor[], Tensor>)model[20]).forward(new Tensor[] { x, outputs[2] }); // Concat with P5
				var p21 = ((Module<Tensor, Tensor>)model[21]).forward(x);

				var list = ((Module<Tensor[], Tensor[]>)model[22]).forward(new Tensor[] { p15, p18, p21 });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;
			}

		}

		public class Yolov11Segment : Module<Tensor, Tensor[]>
		{
			ModuleList<Module> model;
			private int[] strides;

			public Yolov11Segment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov11Segment))
			{
				float depth_multiple = 0.5f;
				float width_multiple = 0.25f;
				int max_channels = 1024;
				bool useC3k = false;

				switch (yoloSize)
				{
					case YoloSize.n:
						{
							depth_multiple = 0.5f;
							width_multiple = 0.25f;
							max_channels = 1024;
							useC3k = false;
							break;
						}
					case YoloSize.s:
						{
							depth_multiple = 0.5f;
							width_multiple = 0.5f;
							max_channels = 1024;
							useC3k = false;
							break;
						}
					case YoloSize.m:
						{
							depth_multiple = 0.5f;
							width_multiple = 1.0f;
							max_channels = 512;
							useC3k = true;
							break;
						}
					case YoloSize.l:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.0f;
							max_channels = 512;
							useC3k = true;
							break;
						}
					case YoloSize.x:
						{
							depth_multiple = 1.0f;
							width_multiple = 1.5f;
							max_channels = 768;
							useC3k = true;
							break;
						}
				}

				int widthSize64 = Math.Min((int)(64 * width_multiple), max_channels);
				int widthSize128 = Math.Min((int)(128 * width_multiple), max_channels);
				int widthSize256 = Math.Min((int)(256 * width_multiple), max_channels);
				int widthSize512 = Math.Min((int)(512 * width_multiple), max_channels);
				int widthSize1024 = Math.Min((int)(1024 * width_multiple), max_channels);
				int depthSize2 = (int)(2 * depth_multiple);

				int[] ch = new int[] { widthSize256, widthSize512, widthSize1024 };
				strides = ch;

				model = new ModuleList<Module>(
					new Conv(3, widthSize64, 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
					new Conv(widthSize64, widthSize128, 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
					new C3k2(widthSize128, widthSize256, depthSize2, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
					new C3k2(widthSize256, widthSize512, depthSize2, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
					new C3k2(widthSize512, widthSize512, depthSize2, c3k: true, device: device, dtype: dtype),
					new Conv(widthSize512, widthSize1024, 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
					new C3k2(widthSize1024, widthSize1024, depthSize2, c3k: true, device: device, dtype: dtype),
					new SPPF(widthSize1024, widthSize1024, 5, device: device, dtype: dtype),                                                          // 9
					new C2PSA(widthSize1024, widthSize1024, depthSize2, device: device, dtype: dtype),                                                //10

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new C3k2(widthSize1024 + widthSize512, widthSize512, depthSize2, useC3k, device: device, dtype: dtype),                           // 13

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new C3k2(widthSize512 + widthSize512, widthSize256, depthSize2, useC3k, device: device, dtype: dtype),                            // 16 (P3/8-small)

					new Conv(widthSize256, widthSize256, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P4
					new C3k2(widthSize512 + widthSize256, widthSize512, depthSize2, useC3k, device: device, dtype: dtype),                            // 19 (P4/16-medium)

					new Conv(widthSize512, widthSize512, 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P5
					new C3k2(widthSize1024 + widthSize512, widthSize1024, depthSize2, c3k: true, device: device, dtype: dtype),                       // 22 (P5/32-large)

					new Segment(ch, nc, npr: widthSize256, legacy: false, device: device, dtype: dtype)                                                // Detect(P3, P4, P5)
					);
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor x)
			{
				using var _ = NewDisposeScope();
				List<Tensor> outputs = new List<Tensor>();
				for (int i = 0; i < 11; i++)
				{
					var md = ((Module<Tensor, Tensor>)model[i]);
					x = md.forward(x);
					if (i == 4 || i == 6 || i == 10)
					{
						outputs.Add(x);
					}
				}

				x = ((Module<Tensor, Tensor>)model[11]).forward(x);
				x = ((Module<Tensor[], Tensor>)model[12]).forward(new Tensor[] { x, outputs[1] });
				Tensor p13 = ((Module<Tensor, Tensor>)model[13]).forward(x);

				x = ((Module<Tensor, Tensor>)model[14]).forward(p13);
				x = ((Module<Tensor[], Tensor>)model[15]).forward(new Tensor[] { x, outputs[0] });
				Tensor p16 = ((Module<Tensor, Tensor>)model[16]).forward(x);

				x = ((Module<Tensor, Tensor>)model[17]).forward(p16);
				x = ((Module<Tensor[], Tensor>)model[18]).forward(new Tensor[] { x, p13 });
				Tensor p19 = ((Module<Tensor, Tensor>)model[19]).forward(x);

				x = ((Module<Tensor, Tensor>)model[20]).forward(p19);
				x = ((Module<Tensor[], Tensor>)model[21]).forward(new Tensor[] { x, outputs[2] });
				Tensor p22 = ((Module<Tensor, Tensor>)model[22]).forward(x);

				var list = ((Module<Tensor[], Tensor[]>)model[23]).forward(new Tensor[] { p16, p19, p22 });
				for (int i = 0; i < list.Length; i++)
				{
					list[i] = list[i].MoveToOuterDisposeScope();
				}
				return list;
			}
		}

	}
}

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	public class Loss
	{
		/// <summary>
		/// Returns label smoothing BCE targets for reducing overfitting
		/// </summary>
		/// <param name="eps"></param>
		/// <returns>pos: `1.0 - 0.5*eps`, neg: `0.5*eps`.</returns>
		static private (float, float) Smooth_BCE(float eps = 0.1f)
		{
			// For details see https://github.com/ultralytics/yolov3/issues/238;  //issuecomment-598028441"""
			return (1.0f - 0.5f * eps, 0.5f * eps);
		}

		private class BCEBlurWithLogitsLoss : Module<Tensor, Tensor, Tensor>
		{
			BCEWithLogitsLoss loss_fcn;
			float alpha;
			public BCEBlurWithLogitsLoss(float alpha = 0.05f, Reduction reduction = Reduction.None) : base("BCEBlurWithLogitsLoss")
			{
				loss_fcn = BCEWithLogitsLoss(reduction: reduction);  // must be nn.BCEWithLogitsLoss()
				this.alpha = alpha;
			}

			public override Tensor forward(Tensor pred, Tensor t)
			{
				var loss = loss_fcn.forward(pred, t);
				pred = sigmoid(pred);  // prob from logits
				var dx = pred - t;// ;  // reduce only missing label effects
								  // dx = (pred - true).abs()  ;  // reduce missing label and false label effects
				var alpha_factor = 1 - exp((dx - 1) / (alpha + 1e-4));
				loss *= alpha_factor;
				return loss.mean();
			}
		}

		private class FocalLoss : Module<Tensor, Tensor, Tensor>
		{
			private readonly BCEWithLogitsLoss loss_fcn;
			private readonly float alpha;
			private readonly float gamma;
			private Reduction reduction;
			public FocalLoss(BCEWithLogitsLoss loss_fcn, float gamma = 1.5f, float alpha = 0.25f) : base("FocalLoss")
			{
				this.loss_fcn = loss_fcn;  // must be nn.BCEWithLogitsLoss()
				this.gamma = gamma;
				this.alpha = alpha;
				reduction = loss_fcn.reduction;
			}

			public override Tensor forward(Tensor pred, Tensor t)
			{
				Tensor loss = loss_fcn.forward(pred, t);
				var pred_prob = sigmoid(pred);  // prob from logits
				var p_t = true * pred_prob + (1 - t) * (1 - pred_prob);

				var alpha_factor = t * alpha + (1 - t) * (1 - alpha);

				var modulating_factor = (1.0 - p_t).pow(gamma);

				loss *= alpha_factor * modulating_factor;

				switch (reduction)
				{
					case Reduction.Mean: return loss.mean();
					case Reduction.Sum: return loss.sum();
					case Reduction.None: return loss;
					default: return loss;
				}

			}
		}


		private class DFLoss : Module<Tensor, Tensor, Tensor>
		{
			private readonly int reg_max;

			public DFLoss(int reg_max = 16) : base("dfloss")
			{
				this.reg_max = reg_max;
			}

			public override Tensor forward(Tensor pred_dist, Tensor target)
			{
				target = target.clamp_(0, this.reg_max - 1 - 0.01);

				var tl = target.@long(); // target left
				var tr = tl + 1; //target right
				var wl = tr - target; //weight left
				var wr = 1 - wl; //weight right
				return (
					functional.cross_entropy(pred_dist, tl.view(-1), reduction: Reduction.None).view(tl.shape) * wl
					+ functional.cross_entropy(pred_dist, tr.view(-1), reduction: Reduction.None).view(tl.shape) * wr
				).mean([-1], keepdim: true);
			}
		}



		private class BboxLoss : Module
		{
			private readonly DFLoss dflLoss;
			private readonly int reg_max;

			public BboxLoss(int regMax = 16) : base("BboxLoss")
			{
				dflLoss = regMax > 1 ? new DFLoss(regMax) : null;
				reg_max = regMax;
			}

			public (Tensor lossIou, Tensor lossDfl) forward(Tensor predDist, Tensor predBboxes, Tensor anchorPoints, Tensor targetBboxes, Tensor targetScores, Tensor targetScoresSum, Tensor fgMask)
			{
				// Step 1: Compute weight
				var weight = targetScores.sum(new long[] { -1 })[fgMask].unsqueeze(-1);

				// Step 2: Compute IoU
				var iou = bbox_iou(predBboxes[fgMask], targetBboxes[fgMask], false, true);
				var lossIou = ((1.0 - iou) * weight).sum() / targetScoresSum;

				// Step 3: Compute DFL loss
				Tensor lossDfl;
				if (dflLoss != null)
				{
					var targetLtrb = bbox2dist(anchorPoints, targetBboxes, reg_max - 1);
					lossDfl = dflLoss.forward(predDist[fgMask].view(-1, reg_max), targetLtrb[fgMask]) * weight;
					lossDfl = lossDfl.sum() / targetScoresSum;
				}
				else
				{
					lossDfl = torch.tensor(0.0, device: predDist.device);
				}

				return (lossIou, lossDfl);
			}

			private Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true, bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7f)
			{
				Tensor b1_x1, b1_x2, b1_y1, b1_y2;
				Tensor b2_x1, b2_x2, b2_y1, b2_y2;
				Tensor w1, h1, w2, h2;

				if (xywh)  // transform from xywh to xyxy
				{
					Tensor[] xywh1 = box1.chunk(4, -1);
					Tensor x1 = xywh1[0];
					Tensor y1 = xywh1[1];
					w1 = xywh1[2];
					h1 = xywh1[3];

					Tensor[] xywh2 = box2.chunk(4, -1);
					Tensor x2 = xywh2[0];
					Tensor y2 = xywh2[1];
					w2 = xywh2[2];
					h2 = xywh2[3];

					var (w1_, h1_, w2_, h2_) = (w1 / 2, h1 / 2, w2 / 2, h2 / 2);
					(b1_x1, b1_x2, b1_y1, b1_y2) = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_);
					(b2_x1, b2_x2, b2_y1, b2_y2) = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_);
				}

				else  // x1, y1, x2, y2 = box1
				{
					Tensor[] b1x1y1x2y2 = box1.chunk(4, -1);
					b1_x1 = b1x1y1x2y2[0];
					b1_y1 = b1x1y1x2y2[1];
					b1_x2 = b1x1y1x2y2[2];
					b1_y2 = b1x1y1x2y2[3];

					Tensor[] b2x1y1x2y2 = box2.chunk(4, -1);
					b2_x1 = b2x1y1x2y2[0];
					b2_y1 = b2x1y1x2y2[1];
					b2_x2 = b2x1y1x2y2[2];
					b2_y2 = b2x1y1x2y2[3];

					(w1, h1) = (b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps));
					(w2, h2) = (b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps));
				}

				// Intersection area
				var inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0);

				// Union Area
				var union = w1 * h1 + w2 * h2 - inter + eps;

				// IoU
				var iou = inter / union;
				if (CIoU || DIoU || GIoU)
				{
					var cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1);  //convex (smallest enclosing box) width
					var ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1);  // convex height
					if (CIoU || DIoU)  // Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
					{
						var c2 = cw.pow(2) + ch.pow(2) + eps;   //convex diagonal squared
						var rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4;   //center dist ** 2

						if (CIoU)  // https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
						{
							var v = 4 / (MathF.PI * MathF.PI) * (atan(w2 / h2) - atan(w1 / h1)).pow(2);
							using (no_grad())
							{
								var alpha = v / (v - iou + (1 + eps));
								return iou - (rho2 / c2 + v * alpha);  //CIoU
							}
						}
						return iou - rho2 / c2;  // DIoU
					}
					var c_area = cw * ch + eps;    // convex area
					return iou - (c_area - union) / c_area;  // GIoU https://arxiv.org/pdf/1902.09630.pdf
				}
				return iou; //IoU
			}

			public Tensor bbox2dist(Tensor anchor_points, Tensor bbox, int reg_max)
			{
				Tensor[] x1y1x2y2 = bbox.chunk(2, -1);
				Tensor x1y1 = x1y1x2y2[0];
				Tensor x2y2 = x1y1x2y2[1];
				return torch.cat([anchor_points - x1y1, x2y2 - anchor_points], -1).clamp_(0, reg_max - 0.01);  // dist (lt, rb)

			}
		}


		public class Yolov5DetectionLoss : Module<Tensor[], Tensor, (Tensor, Tensor)>
		{
			private readonly float lambda_coord = 5.0f;
			private readonly float lambda_noobj = 0.5f;
			private readonly float cp;
			private readonly float cn;
			private float[] balance;
			private readonly int ssi;
			private readonly float gr;
			private readonly bool autobalance;
			private readonly int na;
			private readonly int nc;
			private readonly int nl;
			private readonly float[][] anchors;

			private Device device = new Device(DeviceType.CPU);
			private ScalarType dtype = ScalarType.Float32;

			private readonly float anchor_t = 4.0f;
			private readonly bool sort_obj_iou = false;
			private readonly float h_box = 0.05f;
			private readonly float h_obj = 1.0f;
			private readonly float h_cls = 0.5f;
			private readonly float h_cls_pw = 1.0f;
			private readonly float h_obj_pw = 1.0f;
			private readonly float fl_gamma = 0.0f;
			private readonly float h_label_smoothing = 0.0f;

			public Yolov5DetectionLoss(int nc = 80, bool autobalance = false) : base("Yolov5DetectionLoss")
			{
				int model_nl = 3;
				int[] model_stride = [8, 16, 32];
				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] anchors = [[10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d],
						[30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d],
						[116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d]];

				(cp, cn) = Smooth_BCE(h_label_smoothing);
				balance = model_nl == 3 ? [4.0f, 1.0f, 0.4f] : [4.0f, 1.0f, 0.25f, 0.06f, 0.02f];
				ssi = autobalance ? model_stride.ToList().IndexOf(16) : 0;
				gr = 1.0f;
				this.autobalance = autobalance;

				nl = anchors.Length;
				na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				this.nc = nc; // number of classes
				this.anchors = anchors;
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets)
			{
				device = targets.device;
				dtype = targets.dtype;

				var BCEcls = BCEWithLogitsLoss(pos_weights: tensor(new float[] { h_cls_pw }, device: device));
				var BCEobj = BCEWithLogitsLoss(pos_weights: tensor(new float[] { h_obj_pw }, device: device));

				//var BCEcls = new FocalLoss(BCEWithLogitsLoss(pos_weights: torch.tensor(new float[] { h_cls_pw }, device: this.device)), fl_gamma);
				//var BCEobj = new FocalLoss(BCEWithLogitsLoss(pos_weights: torch.tensor(new float[] { h_obj_pw }, device: this.device)), fl_gamma);

				var lcls = zeros(1, device: device, dtype: dtype);  // class loss
				var lbox = zeros(1, device: device, dtype: dtype);  // box loss
				var lobj = zeros(1, device: device, dtype: dtype);  // object loss

				var (tcls, tbox, indices, anchors) = build_targets(preds, targets);
				Tensor tobj = zeros(0);
				for (int i = 0; i < preds.Length; i++)
				{
					var pi = preds[i].clone();
					var b = indices[i][0];
					var a = indices[i][1];
					var gj = indices[i][2];
					var gi = indices[i][3];
					tobj = zeros(preds[i].shape.Take(4).ToArray(), device: device, dtype: dtype);  // targets obj
					long n = b.shape[0];
					if (n > 0)
					{
						var temp = pi[b, a, gj, gi].split([2, 2, 1, nc], 1);
						var pxy = temp[0];
						var pwh = temp[1];
						var pcls = temp[3];

						pxy = pxy.sigmoid() * 2 - 0.5f;
						pwh = (pwh.sigmoid() * 2).pow(2) * anchors[i];
						var pbox = cat([pxy, pwh], 1);  // predicted box
						var iou = bbox_iou(pbox, tbox[i], CIoU: true).squeeze();  // iou(prediction, targets)
						lbox += (1.0f - iou).mean();  // iou loss

						// Objectness
						iou = iou.detach().clamp(0).type(tobj.dtype);

						if (sort_obj_iou)
						{
							var j = iou.argsort();
							(b, a, gj, gi, iou) = (b[j], a[j], gj[j], gi[j], iou[j]);
						}
						if (gr < 1)
						{
							iou = 1.0f - gr + gr * iou;
						}

						tobj[b, a, gj, gi] = iou;   // iou ratio

						// Classification
						if (nc > 1)  // cls loss (only if multiple classes)
						{
							var tt = full_like(pcls, cn, device: device, dtype: ScalarType.Float32);  // targets
							tt[arange(n), tcls[i]] = cp;
							lcls += BCEcls.forward(pcls, tt.to(dtype));  // BCE
						}

					}

					var obji = BCEobj.forward(pi[TensorIndex.Ellipsis, 4], tobj);
					lobj += obji * balance[i];  // obj loss
					if (autobalance)
					{
						balance[i] = balance[i] * 0.9999f + 0.0001f / obji.detach().item<float>();
					}


				}
				if (autobalance)
				{
					for (int i = 0; i < balance.Length; i++)
					{
						balance[i] = balance[i] / balance[ssi];
					}
				}

				lbox *= h_box;

				lobj *= h_obj;

				lcls *= h_cls;

				long bs = tobj.shape[0];  // batch size


				return ((lbox + lobj + lcls) * bs, cat([lbox, lobj, lcls]).detach());
			}

			//已经检查OK
			private (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<Tensor>) build_targets(Tensor[] p, Tensor targets)
			{
				var tcls = new List<Tensor>();
				var tbox = new List<Tensor>();
				var indices = new List<List<Tensor>>();
				var anch = new List<Tensor>();
				int na = this.na;
				int nt = (int)targets.shape[0];  // number of anchors, targets
												 //tcls, tbox, indices, anch = [], [], [], []
				var gain = ones(7, device: device, dtype: dtype);// normalized to gridspace gain
				var ai = arange(na, device: device, dtype: dtype).view(na, 1).repeat(1, nt);  // same as .repeat_interleave(nt)
				targets = cat([targets.repeat(na, 1, 1), ai.unsqueeze(-1)], 2);// append anchor indices


				float g = 0.5f;  // bias
				var off = tensor(new int[,] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, device: device) * g;
				for (int i = 0; i < nl; i++)
				{
					Tensor anchors = this.anchors[i];
					anchors = anchors.view(3, 2).to(dtype, device);
					var shape = p[i].shape;
					var temp = tensor(new float[] { shape[3], shape[2], shape[3], shape[2] }, device: device, dtype: dtype);

					gain.index_put_(temp, new long[] { 2, 3, 4, 5 });
					var t = targets * gain;
					Tensor offsets = zeros(0, device: device);
					if (nt != 0)
					{
						var r = t[TensorIndex.Ellipsis, TensorIndex.Slice(4, 6)] / anchors.unsqueeze(1);
						var j = max(r, 1 / r).max(2).values < anchor_t;  // compare
						t = t[j];  //filter
						var gxy = t[TensorIndex.Ellipsis, TensorIndex.Slice(2, 4)];   // grid xy
						var gxi = gain[TensorIndex.Ellipsis, TensorIndex.Slice(2, 4)] - gxy; // inverse
						Tensor jk = (gxy % 1 < g & gxy > 1).T;
						j = jk[0];
						var k = jk[1];
						Tensor lm = (gxi % 1 < g & gxi > 1).T;
						var l = lm[0];
						var m = lm[1];
						j = stack([ones_like(j), j, k, l, m]);
						t = t.repeat([5, 1, 1])[j];
						offsets = (zeros_like(gxy).unsqueeze(0) + off.unsqueeze(1))[j];
					}
					else
					{
						t = targets[0];
						offsets = zeros(1);
					}

					Tensor[] ck = t.chunk(4, 1); // (image, class), grid xy, grid wh, anchors
					var bc = ck[0];
					var gxy_ = ck[1];
					var gwh = ck[2];
					var a = ck[3];

					a = a.@long().view(-1);
					bc = bc.@long().T; // anchors, image, class
					Tensor b = bc[0];
					Tensor c = bc[1];

					var gij = (gxy_ - offsets).@long();// grid indices
					var gi = gij.T[0];
					var gj = gij.T[1];

					indices.Add(new List<Tensor> { b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1) });// image, anchor, grid
					tbox.Add(cat([gxy_ - gij, gwh], 1));  // box
					anch.Add(anchors[a]); // anchors
					tcls.Add(c);// class
				}
				return (tcls, tbox, indices, anch);

			}

			//已经检查OK
			private Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true, bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7f)
			{
				Tensor b1_x1, b1_x2, b1_y1, b1_y2;
				Tensor b2_x1, b2_x2, b2_y1, b2_y2;
				Tensor w1, h1, w2, h2;

				if (xywh)  // transform from xywh to xyxy
				{
					Tensor[] xywh1 = box1.chunk(4, -1);
					Tensor x1 = xywh1[0];
					Tensor y1 = xywh1[1];
					w1 = xywh1[2];
					h1 = xywh1[3];

					Tensor[] xywh2 = box2.chunk(4, -1);
					Tensor x2 = xywh2[0];
					Tensor y2 = xywh2[1];
					w2 = xywh2[2];
					h2 = xywh2[3];

					var (w1_, h1_, w2_, h2_) = (w1 / 2, h1 / 2, w2 / 2, h2 / 2);
					(b1_x1, b1_x2, b1_y1, b1_y2) = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_);
					(b2_x1, b2_x2, b2_y1, b2_y2) = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_);
				}

				else  // x1, y1, x2, y2 = box1
				{
					Tensor[] b1x1y1x2y2 = box1.chunk(4, -1);
					b1_x1 = b1x1y1x2y2[0];
					b1_y1 = b1x1y1x2y2[1];
					b1_x2 = b1x1y1x2y2[2];
					b1_y2 = b1x1y1x2y2[3];

					Tensor[] b2x1y1x2y2 = box2.chunk(4, -1);
					b2_x1 = b2x1y1x2y2[0];
					b2_y1 = b2x1y1x2y2[1];
					b2_x2 = b2x1y1x2y2[2];
					b2_y2 = b2x1y1x2y2[3];

					(w1, h1) = (b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps));
					(w2, h2) = (b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps));
				}

				// Intersection area
				var inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0);

				// Union Area
				var union = w1 * h1 + w2 * h2 - inter + eps;

				// IoU
				var iou = inter / union;
				if (CIoU || DIoU || GIoU)
				{
					var cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1);  //convex (smallest enclosing box) width
					var ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1);  // convex height
					if (CIoU || DIoU)  // Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
					{
						var c2 = cw.pow(2) + ch.pow(2) + eps;   //convex diagonal squared
						var rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4;   //center dist ** 2

						if (CIoU)  // https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
						{
							var v = 4 / (MathF.PI * MathF.PI) * (atan(w2 / h2) - atan(w1 / h1)).pow(2);
							using (no_grad())
							{
								var alpha = v / (v - iou + (1 + eps));
								return iou - (rho2 / c2 + v * alpha);  //CIoU
							}
						}
						return iou - rho2 / c2;  // DIoU
					}
					var c_area = cw * ch + eps;    // convex area
					return iou - (c_area - union) / c_area;  // GIoU https://arxiv.org/pdf/1902.09630.pdf
				}
				return iou; //IoU
			}
		}

		public class YolovDetectionLoss : Module<Tensor[], Tensor, (Tensor, Tensor)>
		{
			private readonly int[] stride;
			private readonly int nc;
			private readonly int no;
			private readonly int reg_max;
			private readonly int tal_topk;
			private Device device;
			private readonly bool use_dfl;

			private readonly BCEWithLogitsLoss bce;

			private readonly float hyp_box = 7.5f;
			private readonly float hyp_cls = 0.5f;
			private readonly float hyp_dfl = 1.5f;


			public YolovDetectionLoss(int nc = 80, int reg_max = 16, int tal_topk = 10) : base("YolovDetectionLoss")
			{
				this.stride = [8, 16, 32];
				this.bce = BCEWithLogitsLoss(reduction: Reduction.None);
				this.nc = nc; // number of classes
				this.no = nc + reg_max * 4;
				this.reg_max = reg_max;
				this.use_dfl = reg_max > 1;
				this.tal_topk = tal_topk;
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets)
			{
				using var _ = NewDisposeScope();
				this.device = preds[0].device;
				Tensor loss = torch.zeros(3, device: this.device); // box, cls, dfl
				Tensor[] feats = (Tensor[])preds.Clone();
				List<Tensor> feats_mix = new List<Tensor>();
				foreach (Tensor xi in feats)
				{
					feats_mix.Add(xi.view(feats[0].shape[0], this.no, -1));
				}
				Tensor[] pred_distri_scores = torch.cat(feats_mix, 2).split([this.reg_max * 4, this.nc], 1);
				Tensor pred_distri = pred_distri_scores[0];
				Tensor pred_scores = pred_distri_scores[1];

				pred_scores = pred_scores.permute(0, 2, 1).contiguous();
				pred_distri = pred_distri.permute(0, 2, 1).contiguous();

				ScalarType dtype = pred_scores.dtype;
				long batch_size = pred_scores.shape[0];

				Tensor imgsz = torch.tensor(feats[0].shape[2..], device: this.device, dtype: dtype) * this.stride[0]; // image size (h,w)
				var (anchor_points, stride_tensor) = make_anchors(feats, this.stride, 0.5f);
				var indices = torch.tensor(new long[] { 1, 0, 1, 0 }, device: device);

				// Select elements from imgsz
				var scale_tensor = torch.index_select(imgsz, 0, indices).to(device);
				targets = postprocess(targets, batch_size, scale_tensor);
				var gt_labels_bboxes = targets.split([1, 4], 2); // cls, xyxy
				var gt_labels = gt_labels_bboxes[0];
				var gt_bboxes = gt_labels_bboxes[1];
				var mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);

				Tensor pred_bboxes = bbox_decode(anchor_points, pred_distri);  // xyxy, (b, h*w, 4)

				TaskAlignedAssigner assigner = new TaskAlignedAssigner(topk: tal_topk, num_classes: this.nc, alpha: 0.5f, beta: 6.0f);
				var (_, target_bboxes, target_scores, fg_mask, _) = assigner.forward(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt);

				var target_scores_sum = torch.max(target_scores.sum());
				loss[1] = this.bce.forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum;  // BCE
																											 //float loss1 = (this.bce.forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum).ToSingle();  // BCE
				if (fg_mask.sum().ToInt64() > 0)
				{
					target_bboxes /= stride_tensor;
					(loss[0], loss[2]) = new BboxLoss().forward(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask);
				}
				loss[0] *= this.hyp_box;  // box gain
				loss[1] *= this.hyp_cls;// cls gain
				loss[2] *= this.hyp_dfl;// dfl gain
				//return ((loss.sum() * batch_size).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());
				return ((loss.sum()).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());
			}

			private Tensor bbox_decode(Tensor anchor_points, Tensor pred_dist)
			{
				using var _ = NewDisposeScope();
				// Decode predicted object bounding box coordinates from anchor points and distribution.
				Tensor proj = torch.arange(this.reg_max, dtype: pred_dist.dtype, device: pred_dist.device);
				if (this.use_dfl)
				{
					pred_dist = pred_dist.view(pred_dist.shape[0], pred_dist.shape[1], 4, pred_dist.shape[2] / 4).softmax(3).matmul(proj);
				}

				return dist2bbox(pred_dist, anchor_points, xywh: false).MoveToOuterDisposeScope();
			}

			private Tensor dist2bbox(Tensor distance, Tensor anchor_points, bool xywh = true, int dim = -1)
			{
				using var _ = NewDisposeScope();
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
				return torch.cat([x1y1, x2y2], dim).MoveToOuterDisposeScope(); // xyxy bbox
			}

			private (Tensor, Tensor) make_anchors(Tensor[] feats, int[] strides, float grid_cell_offset = 0.5f)
			{
				using var _ = NewDisposeScope();
				ScalarType dtype = feats[0].dtype;
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
				return (torch.cat(anchor_points).MoveToOuterDisposeScope(), torch.cat(stride_tensor).MoveToOuterDisposeScope());
			}

			private Tensor postprocess(Tensor targets, long batch_size, Tensor scale_tensor)
			{
				using var _ = NewDisposeScope();
				// Preprocesses the target counts and matches with the input batch size to output a tensor.
				long nl = targets.shape[0];
				long ne = targets.shape[1];

				if (nl == 0)
				{
					return torch.zeros([batch_size, 0, ne - 1], device: this.device);
				}
				else
				{
					Tensor i = targets[TensorIndex.Colon, 0];  // image index
					var (_, _, counts) = i.unique(return_counts: true);
					Tensor _out = torch.zeros([batch_size, counts.max().ToInt64(), ne - 1], device: this.device);

					for (int j = 0; j < batch_size; j++)
					{
						Tensor matches = i == j;
						long n = matches.sum().ToInt64();
						if (n > 0)
						{
							// Get the indices where matches is True
							var indices = torch.nonzero(matches).squeeze().to(ScalarType.Int64);

							// Select the rows from targets
							var selectedRows = targets.index_select(0, indices.contiguous());

							// Slice the rows to exclude the first column
							var selectedRowsSliced = selectedRows.narrow(1, 1, ne - 1);

							// Assign to the output tensor
							_out[j, TensorIndex.Slice(0, n)] = selectedRowsSliced;
						}

					}
					// Convert xywh to xyxy format and scale
					_out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)] = xywh2xyxy(_out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)].mul(scale_tensor));
					return _out.MoveToOuterDisposeScope();
				}


			}

			private Tensor xywh2xyxy(Tensor xywh)
			{
				using var _ = NewDisposeScope();
				var xyxy = torch.zeros_like(xywh);
				xyxy[TensorIndex.Ellipsis, 0] = xywh[TensorIndex.Ellipsis, 0] - xywh[TensorIndex.Ellipsis, 2] / 2; // x1
				xyxy[TensorIndex.Ellipsis, 1] = xywh[TensorIndex.Ellipsis, 1] - xywh[TensorIndex.Ellipsis, 3] / 2; // y1
				xyxy[TensorIndex.Ellipsis, 2] = xywh[TensorIndex.Ellipsis, 0] + xywh[TensorIndex.Ellipsis, 2] / 2; // x2
				xyxy[TensorIndex.Ellipsis, 3] = xywh[TensorIndex.Ellipsis, 1] + xywh[TensorIndex.Ellipsis, 3] / 2; // y2
				return xyxy.MoveToOuterDisposeScope();
			}

		}

	}



}



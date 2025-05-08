from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin

import pickle, torch

def load_solver(scheduler, p):
  with open(p, 'rb') as f:
    scheduler.train_params = pickle.load(f)
    if isinstance(scheduler.train_params, torch.Tensor):
      scheduler.train_params = scheduler.train_params.half()
    elif isinstance(scheduler.train_params, list):
      for i in range(len(scheduler.train_params)):
        scheduler.train_params[i] = scheduler.train_params[i].half()


@scheduler_registry.add_to_registry("TDEIS")
class TDEIS(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 6:
      timesteps = [999.0, 937.891357421875, 869.486083984375, 775.603271484375, 613.0316162109375, 322.08416748046875]
    elif num_inference_steps == 10:
      timesteps = [999.0, 960.15380859375, 923.48974609375, 885.3140869140625, 844.8024291992188, 784.7818603515625, 700.8524780273438, 547.05419921875, 275.96478271484375, 53.98443603515625]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TUCOEFEXT2")
class TUCOEFEXT2(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 6:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:400,10_nfe:6_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 937.8103637695312, 866.4406127929688, 764.21630859375, 574.0706176757812, 210.02117919921875]
      self.unet_timesteps = [999.0, 938.2570190429688, 883.1891479492188, 775.9857788085938, 570.0113525390625, 211.54656982421875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:400,10_nfe:10_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 957.8643188476562, 919.55517578125, 881.1990356445312, 836.5147094726562, 782.945556640625, 704.312744140625, 556.6346435546875, 295.40911865234375, 58.51824951171875]
      self.unet_timesteps = [999.0, 961.334716796875, 923.9769897460938, 886.8089599609375, 843.2684326171875, 792.1455688476562, 706.0149536132812, 535.715576171875, 246.64202880859375, 54.10955810546875]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("COEF41UP")
class COEF41UP(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 6:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEF41UP_DEIS40_train:400,10_nfe:6_slr:1,1,1_L:lpips_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 919.8023681640625, 822.2236938476562, 692.0716552734375, 461.43206787109375, 100.51776123046875]
      self.unet_timesteps = [999.0, 919.1463623046875, 823.329345703125, 684.0023803710938, 432.25323486328125, 119.0926513671875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEF41UP_DEIS40_train:400,10_nfe:10_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 960.764892578125, 924.0972900390625, 883.2330932617188, 833.9671630859375, 773.0445556640625, 681.6389770507812, 516.8616943359375, 256.02728271484375, 41.5020751953125]
      self.unet_timesteps = [999.0, 958.4864501953125, 922.8566284179688, 878.4332275390625, 835.5626220703125, 774.3377685546875, 680.220703125, 503.30267333984375, 239.97259521484375, 51.47705078125]
    self.prepare_solver_data(timesteps, device)

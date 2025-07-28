from registries import scheduler_registry
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
    if num_inference_steps == 4:
      timesteps = [999.0, 876.153076171875, 692.7985229492188, 404.99066162109375] 
    elif num_inference_steps == 10:
      timesteps = [999.0, 960.15380859375, 923.48974609375, 885.3140869140625, 844.8024291992188, 784.7818603515625, 700.8524780273438, 547.05419921875, 275.96478271484375, 53.98443603515625]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TCOEF")
class TCOEF(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEF_DEIS40_train:50,10_nfe:4_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 862.8033447265625, 650.07763671875, 300.910888671875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEF_DEIS40_train:50,10_nfe:10_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 959.9639892578125, 923.6149291992188, 885.6728515625, 843.1478881835938, 781.790283203125, 694.749267578125, 542.995849609375, 289.14605712890625, 57.32183837890625]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TCOEFEXT")
class TCOEFEXT(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT_DEIS40_train:50,10_nfe:4_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 859.1026611328125, 648.2139892578125, 298.30291748046875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT_DEIS40_train:50,10_nfe:10_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 960.1036987304688, 924.030517578125, 886.4566650390625, 843.4082641601562, 782.1332397460938, 696.097412109375, 546.26904296875, 292.71435546875, 56.710693359375]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TCOEFEXT2")
class TCOEFEXT2(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:50,10_nfe:4_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 864.5635375976562, 654.88671875, 301.17236328125]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:50,10_nfe:10_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 959.6969604492188, 923.3590698242188, 886.4968872070312, 843.8392944335938, 780.26806640625, 691.5890502929688, 535.066650390625, 277.65869140625, 54.83673095703125]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TCOEFEXTF")
class TCOEFEXTF(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXTF_DEIS40_train:50,10_nfe:4_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 860.6580810546875, 647.0660400390625, 298.64617919921875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXTF_DEIS40_train:50,10_nfe:10_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 959.26318359375, 922.1982421875, 881.8723754882812, 839.6376953125, 779.0322265625, 685.8663330078125, 522.4505615234375, 271.4656982421875, 56.36962890625]
    self.prepare_solver_data(timesteps, device)
    
@scheduler_registry.add_to_registry("TCOEF41UP")
class TCOEFUP41_(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFUP*4|1_DEIS40_train:50,10_nfe:4_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 860.1639404296875, 643.1583862304688, 299.64569091796875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFUP*4|1_DEIS40_train:50,10_nfe:10_slr:1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 953.5083618164062, 905.558837890625, 850.1173706054688, 787.7232666015625, 710.422607421875, 599.082763671875, 421.0435791015625, 194.33001708984375, 33.01287841796875]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TUCOEFEXT")
class TUCOEFEXT(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT_DEIS40_train:50,10_nfe:4_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 864.041259765625, 641.5648193359375, 272.74151611328125]
      self.unet_timesteps = [999.0, 840.0362548828125, 635.614990234375, 310.99090576171875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT_DEIS40_train:50,10_nfe:10_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 957.1631469726562, 917.3367309570312, 877.1332397460938, 832.2451171875, 767.3975830078125, 668.6771240234375, 499.75616455078125, 267.9722900390625, 61.166748046875]
      self.unet_timesteps = [999.0, 959.2991943359375, 930.1771850585938, 878.9459838867188, 845.1571655273438, 779.4166259765625, 674.3862915039062, 480.30706787109375, 249.07122802734375, 60.0018310546875]
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("TUCOEFEXT2")
class TUCOEFEXT2(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:50,10_nfe:4_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 857.059814453125, 620.443603515625, 263.6326904296875]
      self.unet_timesteps = [999.0, 821.926513671875, 615.9346923828125, 309.48748779296875]
    elif num_inference_steps == 10:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SDXL_TRAIN_TS_SOLVER/COEFEXT2_DEIS40_train:50,10_nfe:10_slr:1,1,1_L:l1_ts:linear.pkl'
      load_solver(self, p)
      timesteps = [999.0, 957.0469360351562, 916.8922119140625, 876.5738525390625, 825.1163330078125, 754.6287841796875, 646.7246704101562, 466.04779052734375, 251.34234619140625, 60.38433837890625]
      self.unet_timesteps = [999.0, 957.3348999023438, 930.7140502929688, 878.9396362304688, 838.5272827148438, 773.6032104492188, 646.9552001953125, 447.0804443359375, 235.71697998046875, 60.7728271484375]
    self.prepare_solver_data(timesteps, device)





# @scheduler_registry.add_to_registry("STUDENTDEIS40")
# class STUDENTDEIS40Scheduler(SchedulerMixin):
#   def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
#     if num_inference_steps == 4:
#       timesteps = [999.0, 876.153076171875, 692.7985229492188, 404.99066162109375] 
#     # elif num_inference_steps == 5:
#     #   timesteps = [999.0000, 917.7686, 829.6602, 674.2077, 404.3070]
#     # elif num_inference_steps == 6:
#     #   timesteps = [999.0, 919.3284912109375, 835.326171875, 705.3441162109375, 462.67596435546875, 156.604248046875]
#     # elif num_inference_steps == 7:
#     #   timesteps = [999.0000, 959.2513, 910.1037, 844.2115, 751.3869, 586.9683, 291.7593]
#     # elif num_inference_steps == 8:
#     #   timesteps = [999.0, 937.361083984375, 870.4226684570312, 794.496826171875, 692.6321411132812, 510.07373046875, 311.46063232421875, 137.58123779296875]
#     # elif num_inference_steps == 9:
#     #   timesteps = [999.0, 955.4049682617188, 903.2561645507812, 853.7620239257812, 793.8970947265625, 690.4998168945312, 516.7586669921875, 291.1212158203125, 110.97943115234375]
#     elif num_inference_steps == 10:
#       timesteps = [999.0, 960.15380859375, 923.48974609375, 885.3140869140625, 844.8024291992188, 784.7818603515625, 700.8524780273438, 547.05419921875, 275.96478271484375, 53.98443603515625]
#     # elif num_inference_steps == 11: # INTERPOLATED
#     #   timesteps = [999., 962.5420492757673, 927.9297088033413, 894.0649331330787, 858.9187981008736, 807.6408674491402, 733.7280021784874, 628.6791228068817, 465.17836915881225, 249.97744141083967, 86.10949999999995]
#     # elif num_inference_steps == 12: # INTERPOLATED
#     #   timesteps = [999., 965.8006847071149, 934.1171595029213, 903.220277426551, 871.9719592738345, 836.644579152407, 779.6426188941776, 702.6731816241967, 591.1485889464, 419.7291111544657, 226.89429099684912, 86.10949999999995]
#     # elif num_inference_steps == 13: # INTERPOLATED
#     #   timesteps = [999.0, 958.7641462726103, 917.6077051388799, 875.1571834158217, 833.5255878182343, 783.845505598965, 714.95321662464, 607.6529959614642, 472.7925088687497, 343.01889542176366, 228.25551482266002, 126.34944350337373, 38.4559]
#     # elif num_inference_steps == 14:
#     #   timesteps = [999.0000, 961.9507, 924.4011, 884.3823, 848.0552, 805.2089, 754.8856, 677.1332, 562.4333, 433.4813, 317.2736, 213.7070, 120.4547,  38.4559]
#     else:
#       raise NotImplementedError

#     self.prepare_solver_data(timesteps, device)


# @scheduler_registry.add_to_registry("STUDENTDDIM200")
# class STUDENTDDIM200Scheduler(SchedulerMixin):
#   def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
#     if num_inference_steps == 5:
#       timesteps = [999.0000, 916.5345, 834.0232, 680.7986, 407.1682]
#     elif num_inference_steps == 6:
#       timesteps = [999.0, 939.0007934570312, 872.59814453125, 772.748291015625, 591.9019775390625, 239.83050537109375]
#     elif num_inference_steps == 7:
#       timesteps = [999.0000, 964.5082, 928.7987, 876.7093, 801.4840, 667.8565, 395.6930]
#     elif num_inference_steps == 8:
#       timesteps = [999.0, 954.59033203125, 900.2192993164062, 834.8855590820312, 747.2821655273438, 587.4136962890625, 340.78887939453125, 135.790771484375]
#     elif num_inference_steps == 9:
#       timesteps = [999.0, 963.2920532226562, 914.841552734375, 857.4659423828125, 771.4490356445312, 646.3502197265625, 449.7867431640625, 265.22943115234375, 105.75408935546875]
#     elif num_inference_steps == 10:
#       timesteps = [999.0000, 971.8139, 939.7130, 903.7587, 857.7821, 802.9174, 718.3980, 594.1359, 366.2291,  87.9242]
#     elif num_inference_steps == 11: # INTERPOLATED
#       timesteps = [999., 974.4988809032989, 946.0472099736589, 914.3980983754831, 875.885191758638, 829.896483604154, 767.9776259502363, 678.6116618993871, 539.3357466826911, 317.5326642080453, 87.92419999999998]
#     elif num_inference_steps == 12: # INTERPOLATED
#       timesteps = [999., 976.7012002164324, 951.2614899960441, 923.1961564406695, 890.9806259237021, 852.643200170401, 807.75660130861, 740.524465619542, 647.7037250783807, 498.2804119755345, 282.5466920571493, 87.92419999999998]
#     elif num_inference_steps == 13: # INTERPOLATED
#       timesteps = [999., 975.615767030395, 946.5607109341036, 928.3516806572557, 906.338549936765, 881.3136597341788, 848.1045379106279, 802.647989215876, 732.1411221255341, 613.2535489698251, 410.97918650680845, 220.72250140729676, 66.60910000000001]
#     elif num_inference_steps == 14:
#       timesteps = [999.0000, 978.0498, 949.2379, 933.2876, 913.7000, 891.7931, 866.8490, 829.7654, 783.8227, 707.5926, 584.6900, 382.9999, 209.9362,  66.6091]
#     else:
#       raise NotImplementedError

#     self.prepare_solver_data(timesteps, device)
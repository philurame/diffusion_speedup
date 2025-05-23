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

@scheduler_registry.add_to_registry("GS")
class GS(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:4_L:LATENT-L1_S:COEFEXT2_T:IPNDM3_6_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
      timesteps = [999.0, 784.9729614257812, 492.342041015625, 174.104736328125]
      self.unet_timesteps = [999.0, 783.666748046875, 524.8583984375, 216.271484375]

    if num_inference_steps == 5:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:5_L:LATENT-L1_S:COEFEXT2_T:IPNDM3P_7_B:100,10_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
      timesteps = [999.0, 849.1043701171875, 539.812744140625, 302.32952880859375, 129.8460693359375]
      self.unet_timesteps = [999.0, 849.36962890625, 574.4070434570312, 355.1898193359375, 159.9161376953125]

    if num_inference_steps == 6:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:6_L:LATENT-L1_S:COEFEXT2_T:IPNDM3_7_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
      timesteps = [999.0, 851.0889282226562, 546.3758544921875, 349.34942626953125, 231.391357421875, 103.7535400390625]
      self.unet_timesteps = [999.0, 848.7075805664062, 583.421142578125, 400.03057861328125, 257.95599365234375, 114.8118896484375]

    if num_inference_steps == 7:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:7_L:LATENT-L1_S:COEFEXT2_T:UNIPC2HA__8_I-DDIM_lr[2.0|2.0|2.0|2.0].pkl'
      timesteps = [999.0, 817.0244750976562, 564.5325927734375, 417.470947265625, 317.07672119140625, 202.32818603515625, 92.885986328125]
      self.unet_timesteps = [999.0, 815.877685546875, 622.10205078125, 473.41204833984375, 358.2686767578125, 230.8482666015625, 105.57611083984375]
    
    load_solver(self, p)
    self.prepare_solver_data(timesteps, device)

@scheduler_registry.add_to_registry("GAS")
class GAS(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
    if num_inference_steps == 4:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:4_L:LATENT-ADV_S:COEFEXT2_T:IPNDM3_6_I-DDIM_ADV:[0.3|0.1|LL1]_lr[1.0|1.0|1.0|0.3].pkl'
      timesteps = [999.0, 784.18115234375, 492.8530578613281, 169.1993408203125]
      self.unet_timesteps = [999.0, 782.6715087890625, 527.9403076171875, 217.556396484375]

    if num_inference_steps == 5:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:5_L:LATENT-ADV_S:COEFEXT2_T:IPNDM3_7_I-DDIM_ADV:[0.1|0.5|LL1]_lr[1.0|1.0|1.0|0.1].pkl'
      timesteps = [999.0, 854.3948974609375, 558.1051025390625, 321.87542724609375, 137.8121337890625]
      self.unet_timesteps = [999.0, 847.215576171875, 589.0509033203125, 361.14154052734375, 161.74285888671875]

    if num_inference_steps == 6:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:6_L:LATENT-ADV_S:COEFEXT2_T:IPNDM3_7_I-DDIM_ADV:[0.1|0.5|LL1]_lr[1.0|1.0|1.0|0.1].pkl'
      timesteps = [999.0, 853.5631103515625, 567.6004028320312, 375.99395751953125, 244.49261474609375, 107.914794921875]
      self.unet_timesteps = [999.0, 849.724853515625, 596.067138671875, 415.74609375, 273.32379150390625, 121.13946533203125]

    if num_inference_steps == 7:
      p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:7_L:LATENT-ADV_S:COEFEXT2_T:UNIPC2HA__8_I-DDIM_ADV:[1.0|2|LL1]_lr[1.0|1.0|1.0|1.0].pkl'
      timesteps = [999.0, 818.5324096679688, 586.1100463867188, 432.4017333984375, 325.0054931640625, 211.10076904296875, 97.2752685546875]
      self.unet_timesteps = [999.0, 812.3984985351562, 620.705078125, 494.11993408203125, 379.42901611328125, 252.91351318359375, 126.5758056640625]
    
    load_solver(self, p)
    self.prepare_solver_data(timesteps, device)




# # HONEST ONES:
# @scheduler_registry.add_to_registry("GS")
# class GS(SchedulerMixin):
#   def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
#     if num_inference_steps == 4:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:4_L:LATENT-L1_S:COEFEXT2_T:IPNDM3_5_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 767.5469970703125, 410.51055908203125, 166.84173583984375]
#       self.unet_timesteps = [999.0, 767.2385864257812, 443.810791015625, 185.92376708984375]

#     if num_inference_steps == 5:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:5_L:LATENT-L1_S:COEFEXT2_T:IPNDM3_6_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 784.5914306640625, 495.65850830078125, 278.8197021484375, 125.61553955078125]
#       self.unet_timesteps = [999.0, 783.8294677734375, 528.2061767578125, 315.72540283203125, 135.170166015625]

#     if num_inference_steps == 6:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:6_L:LATENT-L1_S:COEFEXT2_T:IPNDM3_7_I-DDIM_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 851.0889282226562, 546.3758544921875, 349.34942626953125, 231.391357421875, 103.7535400390625]
#       self.unet_timesteps = [999.0, 848.7075805664062, 583.421142578125, 400.03057861328125, 257.95599365234375, 114.8118896484375]

#     if num_inference_steps == 7:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:7_L:LATENT-L1_S:COEFEXT2_T:UNIPC2HA__8_I-DDIM_lr[2.0|2.0|2.0|2.0].pkl'
#       timesteps = [999.0, 817.0244750976562, 564.5325927734375, 417.470947265625, 317.07672119140625, 202.32818603515625, 92.885986328125]
#       self.unet_timesteps = [999.0, 815.877685546875, 622.10205078125, 473.41204833984375, 358.2686767578125, 230.8482666015625, 105.57611083984375]
    
#     load_solver(self, p)
#     self.prepare_solver_data(timesteps, device)

# @scheduler_registry.add_to_registry("GAS")
# class GAS(SchedulerMixin):
#   def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):   
#     if num_inference_steps == 4:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:4_L:LATENT-ADV_S:COEFEXT2_T:IPNDM3_5_I-DDIM_ADV:[0.1|0.5|LL1]_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 773.3267211914062, 418.72601318359375, 155.5372314453125]
#       self.unet_timesteps = [999.0, 757.86328125, 425.17938232421875, 172.21905517578125]

#     if num_inference_steps == 5:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:5_L:LATENT-ADV_T:IPNDM3_6_I:DDIM_ADV:[0.1|0.5]_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 786.8468627929688, 508.2291259765625, 305.1619873046875, 144.71258544921875]
#       self.unet_timesteps = [999.0, 785.9417724609375, 538.023681640625, 325.16925048828125, 142.3336181640625]

#     if num_inference_steps == 6:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:6_L:LATENT-ADV_S:COEFEXT2_T:IPNDM3_7_I-DDIM_ADV:[0.1|0.5|LL1]_lr[1.0|1.0|1.0|0.1].pkl'
#       timesteps = [999.0, 853.5631103515625, 567.6004028320312, 375.99395751953125, 244.49261474609375, 107.914794921875]
#       self.unet_timesteps = [999.0, 849.724853515625, 596.067138671875, 415.74609375, 273.32379150390625, 121.13946533203125]

#     if num_inference_steps == 7:
#       p = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS/SD15_TRAIN/FID/nfe:7_L:LATENT-ADV_S:COEFEXT2_T:UNIPC2HA__8_I-DDIM_ADV:[1.0|2|LL1]_lr[1.0|1.0|1.0|1.0].pkl'
#       timesteps = [999.0, 818.5324096679688, 586.1100463867188, 432.4017333984375, 325.0054931640625, 211.10076904296875, 97.2752685546875]
#       self.unet_timesteps = [999.0, 812.3984985351562, 620.705078125, 494.11993408203125, 379.42901611328125, 252.91351318359375, 126.5758056640625]
    
#     load_solver(self, p)
#     self.prepare_solver_data(timesteps, device)
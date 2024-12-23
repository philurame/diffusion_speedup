import torch, os
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity
import time

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from utils.class_registry import ClassRegistry
metrics_registry = ClassRegistry()

class RequiredDataMixin:
  def __init__(self, path_ddim=None, fake_imgs=None, real_imgs=None, real_anns=None,
               pipe=None, nfe=None, verbose=True, batch_size=512, bench_prompt='a photograph of an astronaut riding a horse'):
    self.path_ddim = path_ddim
    self.fake_imgs = fake_imgs
    self.real_imgs = real_imgs
    self.real_anns = real_anns
    self.pipe = pipe
    self.nfe = nfe
    self.verbose = verbose
    self.batch_size = batch_size
    self.bench_prompt = bench_prompt
#####################################################################################################
# TFLOPS & SEC_PER_IMG
#####################################################################################################

@metrics_registry.add_to_registry('TFLOPS')
class TFLOPSMetric(RequiredDataMixin):
  def __init__(self, **data_kwargs):
    RequiredDataMixin.__init__(self, **data_kwargs)

  @torch.inference_mode()
  def __call__(self):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self.pipe(self.bench_prompt, 
             num_inference_steps=self.nfe, 
             guidance_scale=5, 
             return_dict=False, 
             output_type='latent')
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return total_flops/1e12

@metrics_registry.add_to_registry('SEC_PER_IMG')
class SEC_PER_IMG(RequiredDataMixin):
  def __init__(self, **data_kwargs):
    RequiredDataMixin.__init__(self, **data_kwargs)

  @torch.inference_mode()
  def __call__(self, warmup=20, num_gens=50):
    time_total = 0
    n_counted  = 0
    for i in tqdm(range(warmup + num_gens), desc='SEC_PER_IMG...', disable=not self.verbose):
      start_time = time.perf_counter()
      self.pipe(self.bench_prompt, 
             num_inference_steps=self.nfe, 
             guidance_scale=5, 
             generator = torch.Generator(device='cpu').manual_seed(i),
             return_dict=False, 
             output_type='latent')
      torch.cuda.synchronize()
      if i >= warmup:
        time_total += time.perf_counter() - start_time
        n_counted += 1
    return time_total/n_counted

#####################################################################################################
# LPIPS, IS, FID, CLIP
#####################################################################################################

@metrics_registry.add_to_registry('LPIPS')
class LPIPSMetric(LearnedPerceptualImagePatchSimilarity, RequiredDataMixin):
  def __init__(self, **data_kwargs):
    super().__init__(net_type='alex', reduction='mean', normalize=True)
    RequiredDataMixin.__init__(self, **data_kwargs)
    self.decode_ddim = data_kwargs.get('decode_ddim', False)

  @torch.inference_mode()   
  def __call__(self):
    '''
    need better implementation
    
    # non-optimized implementation:
    self.reset()
    if os.path.exists(self.path_ddim):
      ddim_imgs = torch.load(self.path_ddim, map_location='cpu')
      if ddim_imgs.shape[1:] != (3, 1024, 1024):
        if not self.decode_ddim:
          print("DDIM is not decoded, skipping LPIPS")
          return None
        ddim_imgs = decode_vae(self.pipe, ddim_imgs)
      for i in tqdm(range(0, self.fake_imgs.shape[0], self.batch_size), desc='LPIPS...', disable=not self.verbose):
        self.update(self.fake_imgs[i:i+self.batch_size].to(self.device, dtype=torch.float16)/255,
                          ddim_imgs[i:i+self.batch_size].to(self.device, dtype=torch.float16)/255)
      return self.compute().item()
    return None
    '''
    return  None

@metrics_registry.add_to_registry('IS')
class ISMetric(InceptionScore, RequiredDataMixin):
  def __init__(self, **data_kwargs):
    super().__init__(normalize=False)
    RequiredDataMixin.__init__(self, **data_kwargs)

  @torch.inference_mode()
  def __call__(self):
    self.reset()
    for i in tqdm(range(0, self.fake_imgs.shape[0], self.batch_size), desc='IS...', disable=not self.verbose):
      self.update(self.fake_imgs[i:i+self.batch_size].to(self.device))
    return self.compute()[0].item()
  
@metrics_registry.add_to_registry('FID')
class FIDMetric(FrechetInceptionDistance):
  def __init__(self, **data_kwargs):
    super().__init__(feature=2048, normalize=False, reset_real_features=True)
    RequiredDataMixin.__init__(self, **data_kwargs)
  
  @torch.inference_mode()
  def __call__(self):
    self.reset()
    if self.real_imgs is None:
      return None
    for i in tqdm(range(0, self.fake_imgs.shape[0], self.batch_size), desc='FID...', disable=not self.verbose):
      self.update(self.real_imgs[i:i+self.batch_size].to(self.device), real=True)
      self.update(self.fake_imgs[i:i+self.batch_size].to(self.device), real=False)
    return self.compute().item()

@metrics_registry.add_to_registry('CLIP')
class CLIPMetric(CLIPScore, RequiredDataMixin):
  def __init__(self, **data_kwargs):
    super().__init__(model_name_or_path="openai/clip-vit-base-patch32")
    RequiredDataMixin.__init__(self, **data_kwargs)

  @torch.inference_mode()
  def __call__(self):
    self.reset()
    for i in tqdm(range(0, self.fake_imgs.shape[0], self.batch_size), desc='CLIP...', disable=not self.verbose):
      self.update(self.fake_imgs[i:i+self.batch_size].to(self.device), self.real_anns[i:i+self.batch_size])
    return self.compute().item()
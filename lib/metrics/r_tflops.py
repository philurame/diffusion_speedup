from lib.registries import metric_registry
import torch
from torch.profiler import profile, record_function, ProfilerActivity


@metric_registry.add_to_registry('TFLOPS')
class TFLOPS:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    anns = kwargs['anns']
    pipe = kwargs['pipe']
    nfe  = kwargs['nfe']
    timesteps = kwargs.get('timesteps', None)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        _ = pipe(
          prompt = anns[0], 
          num_inference_steps=nfe, 
          timesteps = timesteps,
          guidance_scale=5, 
          output_type='latent'
        )
    total_flops = sum([event.flops for event in prof.key_averages() if event.flops is not None])
    return total_flops/1e12
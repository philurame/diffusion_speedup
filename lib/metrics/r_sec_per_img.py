from lib.registries import metric_registry
import time, torch, tqdm


@metric_registry.add_to_registry('SEC_PER_IMG')
class SEC_PER_IMG:
  @torch.inference_mode()
  def __call__(self, warmup=20, num_gens=50, **kwargs):
    anns = kwargs['anns']
    nfe  = kwargs['nfe']
    pipe = kwargs['pipe']
    timesteps = kwargs.get('timesteps', None)

    time_total = 0
    n_counted  = 0
    for i in tqdm.tqdm(range(warmup + num_gens), desc='SEC_PER_IMG...'):
      start_time = time.perf_counter()
      _ = pipe(
        prompt = anns[i], 
        num_inference_steps=nfe, 
        timesteps=timesteps,
        guidance_scale=5, 
        generator = torch.Generator(device='cpu').manual_seed(i),
        return_dict=False, 
        output_type='latent'
      )
      torch.cuda.synchronize()
      if i >= warmup:
        time_total += time.perf_counter() - start_time
        n_counted += 1
    return time_total/n_counted
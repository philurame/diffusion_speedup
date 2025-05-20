import torch
from diffusers import StableDiffusionPipeline

class BaseSD15(StableDiffusionPipeline):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    half = kwargs.get('half', True)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    pipe = super().from_pretrained(
      "sd-legacy/stable-diffusion-v1-5", 
      cache_dir = "/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/SD15",
      torch_dtype=torch.float16 if half else torch.float32,
      local_files_only=True # use True for HSE cluster
    ).to(device)
    pipe.scheduler_config = {
      "model_path_name": "sd-legacy/stable-diffusion-v1-5",
      "num_train_timesteps": 1000,
      "beta_start": 0.00085,
      "beta_end": 0.012,
      "beta_schedule": "scaled_linear",
      "init_noise_sigma": 1.0
    }
    pipe.is_train = kwargs.get('is_train', False)
    pipe.latent_dims = (4, 64, 64)
    pipe.img_dims    = (3, 512, 512)
    return pipe
  
  def __call__(self, *args, **kwargs):
    if self.is_train:
      return self._call_impl(*args, **kwargs) 
    with torch.no_grad():
      return self._call_impl(*args, **kwargs)
  
  def _call_impl(self,
    prompt: str,
    num_inference_steps: int = 50,
    timesteps = None,
    output_type="latent",
    height = None, 
    width = None,
    **kwargs
    ):
    '''
    returns latents only
    scheduler must be combined with solver first!
    '''
    device = kwargs.get('device') or self._execution_device
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    do_classifier_free_guidance = guidance_scale>0
    generator = kwargs.get("generator", None)

    if not height or not width:
      _is_unet_config_sample_size_int = isinstance(self.unet.config.sample_size, int)
      height = (
        self.unet.config.sample_size if _is_unet_config_sample_size_int else self.unet.config.sample_size[0]
      )
      width = (
        self.unet.config.sample_size if _is_unet_config_sample_size_int else self.unet.config.sample_size[1]
      )
      height, width = height * self.vae_scale_factor, width * self.vae_scale_factor

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
      prompt,
      device,
      num_images_per_prompt=1,
      do_classifier_free_guidance=do_classifier_free_guidance,
      negative_prompt=kwargs.get("negative_prompt", None),
      prompt_embeds=kwargs.get("prompt_embeds", None),
      negative_prompt_embeds=kwargs.get("negative_prompt_embeds", None),
      lora_scale=None,
      clip_skip=None,
    )

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    latents = self.prepare_latents(
      batch_size = batch_size,
      num_channels_latents=self.unet.config.in_channels,
      height=height,
      width=width,
      device=device,
      dtype=prompt_embeds.dtype,
      generator=generator,
      latents=kwargs.get("latents", None),
    ) 

    timesteps, num_inference_steps = self.retrieve_timesteps(
      num_inference_steps, 
      device, 
      timesteps, 
      **kwargs
    )

    if do_classifier_free_guidance:
      prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device)
      self.prompt_embeds = prompt_embeds

    # Diffusion steps
    for i, t in enumerate(timesteps):
      latents = self.make_unet_solver_step(
        self.scheduler, latents, t, guidance_scale, generator=generator,
        added_cond_kwargs={},
        )
    
    if output_type == "latent":
      return latents

    imgs_pt = self.decode_latents(latents)
    
    if output_type == "pt": # returns -1 -> 1
      return imgs_pt * 2 - 1
    
    # if output_type == "img" or anything else return as uint8
    imgs_255 = (imgs_pt*255).clip(0,255).to(device='cpu', dtype=torch.uint8)
    return imgs_255


  def make_unet_solver_step(self, solver, latents, t, guidance_scale, generator, **unet_kwargs):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

    noise_pred = self.unet(
      latent_model_input,
      t,
      timestep_cond=None,
      cross_attention_kwargs=None,
      return_dict=False,
      encoder_hidden_states=self.prompt_embeds,
      added_cond_kwargs=unet_kwargs['added_cond_kwargs'],
    )[0]    

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    new_latents = solver.step(model_output=noise_pred, sample=latents, generator=generator, return_dict=False, latent_dims=self.latent_dims)
    if not isinstance(new_latents, torch.Tensor): new_latents = new_latents[0]
    return new_latents
  

  def retrieve_timesteps(self,num_inference_steps=None, device=None, timesteps=None, **kwargs):
    '''scheduler.set_timesetps(...)'''
    if timesteps is not None:
      self.scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
      timesteps = self.scheduler.timesteps
      num_inference_steps = len(timesteps)
    else:
      self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
      timesteps = self.scheduler.timesteps
    
    if getattr(self.scheduler, 'unet_timesteps', None) is not None:
      timesteps = self.scheduler.unet_timesteps

    return timesteps, num_inference_steps
  
  def decode_latents(self, latents):
    imgs_pt = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
    imgs_pt = self.image_processor.postprocess(imgs_pt, output_type='pt') #do_denormalize=[True]*imgs_pt.shape[0]
    return imgs_pt
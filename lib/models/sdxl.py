import torch
from diffusers import StableDiffusionXLPipeline

class BaseSDXL(StableDiffusionXLPipeline):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    half = kwargs.get('half', True)
    pipe = super().from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", 
      torch_dtype=torch.float16 if half else torch.float32,
      variant="fp16" if half else None,
      local_files_only=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    pipe.scheduler_config = {
      "num_train_timesteps": 1000,
      "beta_start": 0.00085,
      "beta_end": 0.012,
      "beta_schedule": "scaled_linear",
      "init_noise_sigma": 1.0
    }
    pipe.is_train = kwargs.get('is_train', False)
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
    **kwargs
    ):
    '''
    returns latents only
    scheduler must be combined with solver first!
    '''
    device = kwargs.get('device') or self._execution_device
    guidance_scale = kwargs.get('guidance_scale', 5)
    do_classifier_free_guidance = guidance_scale>0

    # Prepare text embeddings
    ( 
      prompt_embeds,
      negative_prompt_embeds,
      pooled_prompt_embeds,
      negative_pooled_prompt_embeds,
    ) = self.encode_prompt(prompt, device=device)

    # generated height, width (should better use 1024 or >=512)
    height = kwargs.get('height', self.default_sample_size * self.vae_scale_factor)
    width  = kwargs.get('width',  self.default_sample_size * self.vae_scale_factor)

    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    # Create initial noise
    latents = self.prepare_latents(
      batch_size = batch_size,
      num_channels_latents=self.unet.config.in_channels,
      height=height,
      width=width,
      dtype=prompt_embeds.dtype,
      device=device,
      generator=kwargs.get("generator", None),
      latents=kwargs.get("latents", None),
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = self.retrieve_timesteps(
      num_inference_steps, 
      device, 
      timesteps, 
    )
    
    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
      text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
      text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
      original_size=kwargs.get('original_size', (height, width)),
      crops_coords_top_left=kwargs.get('crops_coords_top_left', (0,0)),
      target_size=(height, width),
      dtype=prompt_embeds.dtype,
      text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
      prompt_embeds   = torch.cat([negative_prompt_embeds,        prompt_embeds],   dim=0)
      add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
      add_time_ids    = torch.cat([negative_add_time_ids,         add_time_ids],    dim=0)

    prompt_embeds   = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids    = add_time_ids.to(device).repeat(batch_size, 1)   

    # Diffusion steps
    for i, t in enumerate(timesteps):
      # print(latents.sum(), t)
      latents = self.make_unet_solver_step(
        self.scheduler, latents, t, guidance_scale,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
        )
    
    if output_type == "latent":
      return latents
    
    needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
    if needs_upcasting:
      self.upcast_vae()
      latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    latents = latents / self.vae.config.scaling_factor
    imgs = self.vae.decode(latents.to(device), return_dict=False)[0]
    imgs_pt = self.image_processor.postprocess(imgs, output_type='pt')
    
    if needs_upcasting:
      self.vae.to(dtype=torch.float16)
    
    if output_type == "pt": # returns -1 -> 1
      return imgs_pt * 2 - 1

    # if output_type == "img" or anything else return as uint8
    imgs_255 = (imgs_pt*255).clip(0,255).to(device='cpu', dtype=torch.uint8)
    return imgs_255


  def make_unet_solver_step(self, solver, latents, t, guidance_scale, **unet_kwargs):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    # latent_model_input = solver.scale_model_input(latent_model_input, t)

    noise_pred = self.unet(
      latent_model_input,
      t,
      timestep_cond=None,
      cross_attention_kwargs=None,
      return_dict=False,
      encoder_hidden_states=unet_kwargs['encoder_hidden_states'],
      added_cond_kwargs=unet_kwargs['added_cond_kwargs'],
    )[0]

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    new_latents = solver.step(noise_pred, t, latents, return_dict=False)
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
    
    return timesteps, num_inference_steps
  
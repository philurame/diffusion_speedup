try:

  from registries import model_registry
  from lib.models.sora.sora import BaseSora
  import torch
  import torch.utils.checkpoint as cp

  @model_registry.add_to_registry("SORA_TRAIN")
  class TrainSora(BaseSora):
    @classmethod
    def from_pretrained(self, *args, **kwargs):
      kwargs.pop('half', None)
      kwargs.pop('is_train', None)
      kwargs.pop('save_memory', None)
      
      pipe = super().from_pretrained(half=False, is_train=True, save_memory=True, *args, **kwargs)
      for param in pipe.transformer.parameters():
        param.requires_grad = False
      for param in pipe.text_encoder.parameters():
        param.requires_grad = False
      for param in pipe.vae.parameters():
        param.requires_grad = False
      
      pipe.transformer.gradient_checkpointing = True
      pipe.transformer.training = True
      return pipe

    def make_diffusion_solver_step(self, solver, latents, t, guidance_scale, generator, **diffusion_kwargs):
      do_cfg = guidance_scale>0

      latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
      timestep = t.expand(latent_model_input.shape[0])

      # ==================prepare my shape=====================================
      # predict the noise residual
      prompt_embeds = diffusion_kwargs['prompt_embeds']
      prompt_attention_mask = diffusion_kwargs['prompt_attention_mask']
      if prompt_embeds.ndim == 3:
        prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
      if prompt_attention_mask.ndim == 2:
        prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
      
      attention_mask = torch.ones_like(latent_model_input)[:, 0].to(device=diffusion_kwargs['device'])
      # ==================prepare my shape=====================================

      noise_pred = cp.checkpoint(
        self.transformer,
        latent_model_input,
        use_reentrant = False,
        attention_mask=attention_mask, 
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timestep,
        pooled_projections=None,
        return_dict=False,
      )[0]
      assert not torch.any(torch.isnan(noise_pred))

      if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

      noise_pred  = solver.eps_pred_from("v_prediction", noise_pred, latents)
      new_latents = solver.step(model_output=noise_pred, sample=latents, generator=generator, return_dict=False)

      if not isinstance(new_latents, torch.Tensor): new_latents = new_latents[0]
      return new_latents

    def retrieve_timesteps(self,num_inference_steps=None, device=None, timesteps=None, **kwargs):
      '''scheduler.set_timesetps(...)'''
      if timesteps is not None:
        self.scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
      elif num_inference_steps is not None:
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        timesteps = self.scheduler.timesteps

      if kwargs.get('unet_timesteps', None) is not None:
        timesteps = kwargs['unet_timesteps']
      return timesteps, num_inference_steps
      
except:
  pass
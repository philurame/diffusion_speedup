from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL
import torch
import torch.utils.checkpoint as cp


@model_registry.add_to_registry('SDXL_TRAIN')
class TrainSDXL(BaseSDXL):
  @classmethod
  def from_pretrained(self, *args, **kwargs):
    kwargs.pop('half', None)
    kwargs.pop('is_train', None)
    return super().from_pretrained(half=False, is_train=True, *args, **kwargs)

  def make_unet_solver_step(self, solver, latents, t, guidance_scale, **unet_kwargs):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

    noise_pred = cp.checkpoint(
      self.unet,
      latent_model_input,
      t,
      use_reentrant = False,
      timestep_cond=None,
      cross_attention_kwargs=None,
      return_dict=False,
      encoder_hidden_states=unet_kwargs['encoder_hidden_states'],
      added_cond_kwargs=unet_kwargs['added_cond_kwargs'],
    )[0]

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    new_latents = solver.step(noise_pred, latents)
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

from lib.registries import model_registry
from lib.models.sdxl import BaseSDXL
from lib.models.cachers.tgate.tgate_utils import register_forward, tgate_scheduler
import torch

@model_registry.add_to_registry("TGATE")
class TGATE(BaseSDXL):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    '''
    gate_step   (`int` defaults to 10): The time step to stop calculating the cross attention.
    sp_interval (`int` defaults to 5): The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
    fi_interval (`int` defaults to 1): The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
    warm_up     (`int` defaults to 2): The time step to warm up the model inference.
    '''
    pipe = super().from_pretrained()

    pipe.gate_step = kwargs.get('gate_step', 10)
    pipe.sp_interval = kwargs.get('sp_interval', 5)
    pipe.fi_interval = kwargs.get('fi_interval', 1)
    pipe.warm_up = kwargs.get('warm_up', 2)

    return pipe
  
  def make_unet_solver_step(self, solver, latents, t, guidance_scale, **unet_kwargs):
    do_cfg = guidance_scale>0
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

    ############################################################
    # TGATE
    i = self.scheduler.step_index 
    timesteps = self.scheduler.timesteps

    num_warmup_steps = 0
    if i == 0:
      register_forward(self.unet, 
        'Attention',
        ca_kward = {'cache': False,'reuse': False,},
        sa_kward = {'cache': False,'reuse': False,},
        keep_shape=True
      )
    elif i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
      ca_kwards,sa_kwards,keep_shape=tgate_scheduler(
        cur_step=i-num_warmup_steps, 
        gate_step=self.gate_step,
        sp_interval=self.sp_interval,
        fi_interval=self.fi_interval,
        warm_up=self.warm_up
      )

      register_forward(self.unet, 
        'Attention',
        ca_kward=ca_kwards,
        sa_kward=sa_kwards,
        keep_shape=keep_shape
      )
    ############################################################

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
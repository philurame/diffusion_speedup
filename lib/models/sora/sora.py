import torch, sys, os
from transformers import AutoTokenizer, MT5EncoderModel

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

SORA_PATH = os.path.dirname(os.path.abspath(__file__))
DIFF_PATH = os.path.dirname( # DIFFUSION_SPEEDUP
  os.path.dirname( # lib
    os.path.dirname(SORA_PATH) # models
  )
)
FILES_PATH = os.path.join(DIFF_PATH, "DATA", "SORA_FILES")
sys.path.append(os.path.join(SORA_PATH, "Open-Sora-Plan"))

from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper


class BaseSora(DiffusionPipeline):
  model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
  _optional_components = ["text_encoder_2","tokenizer_2","text_encoder","tokenizer"]
  _callback_tensor_inputs = ["latents","prompt_embeds","negative_prompt_embeds","prompt_embeds_2","negative_prompt_embeds_2"]

  def __init__(self, vae, text_encoder, tokenizer, transformer, tokenizer_2, text_encoder_2):
    super().__init__()
    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      transformer=transformer,
      tokenizer_2=tokenizer_2,
      text_encoder_2=text_encoder_2
    )
    
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    p_vae = os.path.join(FILES_PATH, "vae")
    p_text_encoder = os.path.join(FILES_PATH, "text_encoder")
    p_transformer = os.path.join(FILES_PATH, "diffusion_model")
    half = kwargs.pop("half", True)
    device = torch.device(kwargs.pop("device", "cuda"))
    weight_dtype = torch.float16 if half else torch.float32

    vae = ae_wrapper['WFVAEModel_D8_4x8x8'](p_vae)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
    vae.vae_scale_factor = ae_stride_config['WFVAEModel_D8_4x8x8']

    text_encoder_1 = MT5EncoderModel.from_pretrained(
      p_text_encoder, cache_dir=FILES_PATH, 
      torch_dtype=weight_dtype
    ).eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(
      p_text_encoder, cache_dir=FILES_PATH
    )
    transformer_model = OpenSoraT2V_v1_3.from_pretrained(
      p_transformer, cache_dir=FILES_PATH,
      device_map=None, torch_dtype=weight_dtype
    ).eval()

    pipe = cls(vae, text_encoder_1, tokenizer_1, transformer_model, None, None).to(device)
    pipe._device = device

    # if kwargs.get('save_memory', False):
    #   pipe.enable_model_cpu_offload()
    #   pipe.enable_sequential_cpu_offload()
    #   vae.vae.enable_tiling()
    #   vae.vae.t_chunk_enc = 8
    #   vae.vae.t_chunk_dec = vae.vae.t_chunk_enc // 2
  
    pipe.scheduler_config = {
      "model_path_name": "",
      "num_train_timesteps": 1000,
      "beta_start": 0.0001,
      "beta_end": 0.02,
      "beta_schedule": "linear",
      "init_noise_sigma": 1.0
    }
    pipe.is_train = kwargs.get('is_train', False)
    return pipe


  def __call__(self, *args, **kwargs):
    if self.is_train:
      return self._call_impl(*args, **kwargs) 
    with torch.no_grad():
      return self._call_impl(*args, **kwargs)
  
  def _call_impl(
    self,
    prompt: str,
    num_inference_steps = 50,
    timesteps = None,
    num_frames = None,
    height = 352,
    width = 640,
    output_type = "latent",
    max_sequence_length: int = 512,
    **kwargs
    ):

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = kwargs.get('device') or getattr(self, '_device', None) or self.device
    guidance_scale = kwargs.get('guidance_scale', 7.5)
    do_classifier_free_guidance = guidance_scale > 0
        
    num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
    height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
    width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]

    (
      prompt_embeds,
      negative_prompt_embeds,
      prompt_attention_mask,
      negative_prompt_attention_mask,
    ) = self.encode_prompt(
      prompt=prompt,
      device=device,
      dtype=self.transformer.dtype,
      num_samples_per_prompt=kwargs.get('num_samples_per_prompt', 1),
      do_classifier_free_guidance=do_classifier_free_guidance,
      negative_prompt=kwargs.get('negative_prompt', None),
      prompt_embeds=kwargs.get('prompt_embeds', None),
      negative_prompt_embeds=kwargs.get('negative_prompt_embeds', None),
      prompt_attention_mask=kwargs.get('prompt_attention_mask', None),
      negative_prompt_attention_mask=kwargs.get('negative_prompt_attention_mask', None),
      max_sequence_length=max_sequence_length,
      text_encoder_index=0,
    )
      
    timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps, device, timesteps)

    latents = self.prepare_latents(
      batch_size = batch_size * kwargs.get('num_samples_per_prompt', 1),
      num_channels_latents = self.transformer.config.in_channels,
      num_frames = num_frames, 
      height = height,
      width = width,
      dtype = prompt_embeds.dtype,
      device = device,
      generator = kwargs.get("generator", None),
      latents = kwargs.get("latents", None),
    )

    # 7 create image_rotary_emb, style embedding & time ids
    if do_classifier_free_guidance:
      prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
      prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])

    prompt_embeds = prompt_embeds.to(device=device)
    prompt_attention_mask = prompt_attention_mask.to(device=device)


    # 8. Denoising loop
    for i, t in enumerate(timesteps):
      latents = self.make_diffusion_solver_step(
        self.scheduler, latents, t, guidance_scale, kwargs.get("generator", None),
        prompt_embeds = prompt_embeds,
        prompt_attention_mask = prompt_attention_mask,
        device = device
      )


    if output_type == "latent":
      return latents

    videos = self.decode_latents(latents)
    videos = videos[:, :num_frames, :height, :width]

    # Offload all models
    self.maybe_free_model_hooks()

    return videos


  def decode_latents(self, latents):
    print(f'before vae decode {latents.shape}', torch.max(latents).item(), torch.min(latents).item(), torch.mean(latents).item(), torch.std(latents).item())
    video = self.vae.decode(latents.to(self.vae.vae.dtype))
    print(f'after vae decode {latents.shape}', torch.max(video).item(), torch.min(video).item(), torch.mean(video).item(), torch.std(video).item())
    video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
    return video


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

    noise_pred = self.transformer(
      latent_model_input,
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
    else:
      self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
      timesteps = self.scheduler.timesteps
    
    return timesteps, num_inference_steps

  

  # ===============================================================================================================
  # UTILS 
  # ===============================================================================================================
  def encode_prompt(
    self,
    prompt: str,
    device: torch.device = None,
    dtype: torch.dtype = None,
    num_samples_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt = None,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    prompt_attention_mask = None,
    negative_prompt_attention_mask = None,
    max_sequence_length = None,
    text_encoder_index: int = 0,
    ):
    if dtype is None:
      if self.text_encoder_2 is not None:
        dtype = self.text_encoder_2.dtype
      elif self.transformer is not None:
        dtype = self.transformer.dtype
      else:
        dtype = None
    if device is None:
      device = getattr(self, '_execution_device', None) or getattr(self, '_device', None) or torch.device('cuda')


    tokenizers = [self.tokenizer, self.tokenizer_2]
    text_encoders = [self.text_encoder, self.text_encoder_2]

    tokenizer = tokenizers[text_encoder_index]
    text_encoder = text_encoders[text_encoder_index]

    if max_sequence_length is None:
      if text_encoder_index == 0:
        max_length = 512
      if text_encoder_index == 1:
        max_length = 77
    else:
      max_length = max_sequence_length

    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
      text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
      )
      text_input_ids = text_inputs.input_ids

      prompt_attention_mask = text_inputs.attention_mask.to(device)
      prompt_embeds = text_encoder(
        text_input_ids.to(device),
        attention_mask=prompt_attention_mask,
      )
      prompt_embeds = prompt_embeds[0]

      if text_encoder_index == 1:
        prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip

      prompt_attention_mask = prompt_attention_mask.repeat(num_samples_per_prompt, 1)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_samples_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_samples_per_prompt, seq_len, -1)

    if do_classifier_free_guidance and negative_prompt_embeds is None:
      if negative_prompt is None:
        uncond_tokens = [""] * batch_size
      elif prompt is not None and type(prompt) is not type(negative_prompt):
        raise TypeError(
          f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
          f" {type(prompt)}."
        )
      elif isinstance(negative_prompt, str):
        uncond_tokens = [negative_prompt]
      elif batch_size != len(negative_prompt):
        raise ValueError(
          f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
          f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
          " the batch size of `prompt`."
        )
      else:
        uncond_tokens = negative_prompt

      # max_length = prompt_embeds.shape[1]
      uncond_input = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
      )

      negative_prompt_attention_mask = uncond_input.attention_mask.to(device)
      negative_prompt_embeds = text_encoder(
        uncond_input.input_ids.to(device),
        attention_mask=negative_prompt_attention_mask,
      )
      negative_prompt_embeds = negative_prompt_embeds[0]
      if text_encoder_index == 1:
        negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)  # b d -> b 1 d for clip
      negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_samples_per_prompt, 1)

    if do_classifier_free_guidance:
      # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
      seq_len = negative_prompt_embeds.shape[1]

      negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

      negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_samples_per_prompt, 1)
      negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_samples_per_prompt, seq_len, -1)

    return prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask


  def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
    shape = (
      batch_size,
      num_channels_latents,
      (int(num_frames) - 1) // self.vae.vae_scale_factor[0] + 1, 
      int(height) // self.vae.vae_scale_factor[1],
      int(width) // self.vae.vae_scale_factor[2],
    )
    if isinstance(generator, list) and len(generator) != batch_size:
      raise ValueError(
        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
      )
    if latents is None:
      latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
      latents = latents.to(device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
  
  def save_video(self, video, p_to):
    import imageio
    imageio.mimwrite(f'{p_to}.mp4', 
        video.squeeze(),
        fps=18, 
        quality=10
      )

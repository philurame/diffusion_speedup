
from omegaconf import OmegaConf
import torch, os, sys, tqdm

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


HUNYUAN_PATH = os.path.dirname(os.path.abspath(__file__))
DIFF_PATH = os.path.dirname( # DIFFUSION_SPEEDUP
  os.path.dirname( # lib
    os.path.dirname(HUNYUAN_PATH) # models
  )
)
FILES_PATH = os.path.join(DIFF_PATH, "DATA", "HUNYUAN_FILES")
sys.path.append(os.path.join(HUNYUAN_PATH, "HunyuanVideo"))
os.environ["MODEL_BASE"] = os.path.join(FILES_PATH, "ckpts")


from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.modules.fp8_optimization import convert_fp8_linear
from hyvideo.inference import Inference
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D


class BaseHunyuan(DiffusionPipeline):
  model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
  _optional_components = ["text_encoder_2"]
  _exclude_from_cpu_offload = ["transformer"]
  _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

  def __init__(
    self,
    vae,
    text_encoder,
    transformer,
    text_encoder_2 = None,
    args=None,
  ):
      super().__init__()
      self.args = args
      self.register_modules(
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
      )
      self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
      # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


  @classmethod
  def from_pretrained(cls, *args, **kwargs):

    args = {
      # main model + rope
      "model": "HYVideo-T/2-cfgdistill", # "HYVideo-T/2"
      "latent_channels": 16,
      "precision": "bf16", # fp32, fp16, bf16
      "rope_theta": 256,

      # VAE
      "vae": "884-16c-hy",
      "vae_precision": "fp16",
      "vae_tiling": True,
      "text_encoder": "llm",
      "text_encoder_precision": "fp16",
      "text_states_dim": 4096,
      "text_len": 256,
      "tokenizer": "llm",
      "prompt_template": "dit-llm-encode",
      "prompt_template_video": "dit-llm-encode-video",
      "hidden_state_skip_layer": 2,
      "apply_final_norm": False,

      # CLIP
      "text_encoder_2": "clipL",
      "text_encoder_precision_2": "fp16",
      "text_states_dim_2": 768,
      "tokenizer_2": "clipL",
      "text_len_2": 77,

      # # Scheduler
      # "denoise_type": "flow",
      # "flow_shift": 7.0,
      # "flow_reverse": True,
      # "flow_solver": "euler",
      # # "use-linear_quadratic-schedule": True,
      # "linear_schedule-end": 25

      # model loads
      "model_base": f"{FILES_PATH}/ckpts",
      "dit_weight": f"{FILES_PATH}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
      "model_resolution": "540p",
      "load_key": "module",
      "use_cpu_offload": False,

      # inference general?
      "cfg_scale": 1.0,
      "embedded_cfg_scale": 6.0,
      "use_fp8": False,
      "reproduce": False,
      "disable_autocast": False,
    }
    args = OmegaConf.create(args) 
    logger = None

        
    device = torch.device(kwargs.pop("device", "cuda"))

   # =========================== Build main model ===========================
    factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
    in_channels = args.latent_channels
    out_channels = args.latent_channels

    model = load_model(
      args,
      in_channels=in_channels,
      out_channels=out_channels,
      factor_kwargs=factor_kwargs,
    )
    if args.use_fp8:
      convert_fp8_linear(model, args.dit_weight, original_dtype=PRECISION_TO_TYPE[args.precision])
    model = model.to(device)
    model = Inference.load_state_dict(args, model, args.model_base)
    model.eval()
    
    # ============================= Build extra models ========================
    # VAE
    vae, _, s_ratio, t_ratio = load_vae(
      args.vae,
      args.vae_precision,
      logger=logger,
      device=device if not args.use_cpu_offload else "cpu",
    )

    # Text encoder
    if args.prompt_template_video is not None:
      crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    elif args.prompt_template is not None:
      crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    else:
      crop_start = 0
    max_length = args.text_len + crop_start

    # prompt_template
    prompt_template = (
      PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None
    )

    # prompt_template_video
    prompt_template_video = ( 
      PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None
    )

    text_encoder = TextEncoder(
      text_encoder_type=args.text_encoder,
      max_length=max_length,
      text_encoder_precision=args.text_encoder_precision,
      tokenizer_type=args.tokenizer,
      prompt_template=prompt_template,
      prompt_template_video=prompt_template_video,
      hidden_state_skip_layer=args.hidden_state_skip_layer,
      apply_final_norm=args.apply_final_norm,
      reproduce=args.reproduce,
      logger=logger,
      device=device if not args.use_cpu_offload else "cpu",
    )
    text_encoder_2 = None
    if args.text_encoder_2 is not None:
      text_encoder_2 = TextEncoder(
        text_encoder_type=args.text_encoder_2,
        max_length=args.text_len_2,
        text_encoder_precision=args.text_encoder_precision_2,
        tokenizer_type=args.tokenizer_2,
        reproduce=args.reproduce,
        logger=logger,
        device=device if not args.use_cpu_offload else "cpu",
      )


    # if scheduler is None:
    #   if args.denoise_type == "flow":
    #     scheduler = FlowMatchDiscreteScheduler(
    #       shift=args.flow_shift,
    #       reverse=args.flow_reverse,
    #       solver=args.flow_solver,
    #     )
    #   else:
    #       raise ValueError(f"Invalid denoise type {args.denoise_type}")

    pipe = cls(
      vae=vae,
      text_encoder=text_encoder,
      text_encoder_2=text_encoder_2,
      transformer=model,
      args=args,
    )
    pipe.model = model

    if args.use_cpu_offload:
      pipe.enable_sequential_cpu_offload()
    else:
      pipe = pipe.to(device)

    pipe.scheduler_config = {
      "model_path_name": "",
      "num_train_timesteps": 1000,
      "beta_start": 0.0001,
      "beta_end": 0.02,
      "beta_schedule": "linear",
      "init_noise_sigma": 1.0,
      "model_name": "hunyuan",
      "flow_mathing": True,
    }
    pipe.is_train = kwargs.get('is_train', False)
    pipe._device = device

    pipe.latent_dims = (16, 24, 80, 80)
    pipe.img_dims    = (93, 640, 640, 3)

    pipe.max_timestep = 1000.5
      
    return pipe


  @property
  def device(self):
    return self._device
  

  @torch.no_grad()
  def __call__(
    self,
    prompt,
    num_inference_steps = 50,
    timesteps = None,
    sigmas = None,
    num_frames = 93, # video_length
    height = 640,
    width = 640,
    output_type = "video", # latent, video
    **kwargs
    ):

    data_type = kwargs.get("data_type", "video")
    target_height = align_to(height, 16)
    target_width = align_to(width, 16)
    target_video_length = num_frames
    freqs_cis = self.get_rotary_pos_embed(
      target_video_length, target_height, target_width
    )
    
    if isinstance(prompt, str):
      prompt = [prompt.strip()]

    batch_size = len(prompt)
    guidance_scale = kwargs.get('guidance_scale', 1)
    do_classifier_free_guidance = guidance_scale > 1
    embedded_guidance_scale = self.args.embedded_cfg_scale

    negative_prompt = kwargs.get('negative_prompt', NEGATIVE_PROMPT)
    if guidance_scale == 1.0: negative_prompt = ""
    if not isinstance(negative_prompt, str):
      raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")
    negative_prompt = [negative_prompt.strip()]

    (
      prompt_embeds,
      negative_prompt_embeds,
      prompt_mask,
      negative_prompt_mask,
    ) = self.encode_prompt(
      prompt,
      self.device,
      do_classifier_free_guidance,
      negative_prompt,
      prompt_embeds=kwargs.get('prompt_embeds', None),
      attention_mask=kwargs.get('attention_mask', None),
      negative_prompt_embeds=kwargs.get('negative_prompt_embeds', None),
      negative_attention_mask=kwargs.get('negative_prompt_attention_mask', None),
      data_type=data_type,
    )
    if self.text_encoder_2 is not None:
      (
        prompt_embeds_2,
        negative_prompt_embeds_2,
        prompt_mask_2,
        negative_prompt_mask_2,
      ) = self.encode_prompt(
        prompt,
        self.device,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=None,
        attention_mask=None,
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        text_encoder=self.text_encoder_2,
        data_type=data_type,
      )
    else:
      prompt_embeds_2 = None
      negative_prompt_embeds_2 = None
      prompt_mask_2 = None
      negative_prompt_mask_2 = None


    if do_classifier_free_guidance:
      prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
      if prompt_mask is not None:
        prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
      if prompt_embeds_2 is not None:
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
      if prompt_mask_2 is not None:
        prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])


    timesteps, num_inference_steps = self.retrieve_timesteps(num_inference_steps, timesteps, sigmas, self.device, **kwargs)


    vae_ver = kwargs.get("vae_ver", self.args.vae)
    if "884" in vae_ver: 
      num_frames = (num_frames - 1) // 4 + 1
    elif "888" in vae_ver: 
      num_frames = (num_frames - 1) // 8 + 1
    else: 
      num_frames = num_frames

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
      batch_size * 1,
      num_channels_latents,
      height,
      width,
      num_frames,
      prompt_embeds.dtype,
      device=self.device,
      generator=kwargs.get("generator", None),
      latents=kwargs.get("latents", None),
    )


    target_dtype = PRECISION_TO_TYPE[self.args.precision]
    autocast_enabled = (target_dtype != torch.float32) and not self.args.disable_autocast
    # vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
    # vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast

    # print(vae_ver, guidance_scale, num_channels_latents, latents.shape)
    # print(-1, self.args)

    # 8. Denoising loop
    for t in tqdm.tqdm(timesteps, disable=not kwargs.get("verbose", False)):
      latents = self.make_diffusion_solver_step(
        self.scheduler, latents, t, guidance_scale, kwargs.get("generator", None),
        prompt_embeds = prompt_embeds,
        prompt_embeds_2 = prompt_embeds_2,
        prompt_mask = prompt_mask, 
        freqs_cis = freqs_cis,
        embedded_guidance_scale = embedded_guidance_scale,
        target_dtype = target_dtype,
        device = self.device,
        autocast_enabled=autocast_enabled
      )


    if output_type == "latent":
      return latents
  
    videos = self.decode_latents(latents, self.args.vae_tiling, cpu=True)

    # Offload all models
    self.maybe_free_model_hooks()

    return videos


  @torch.no_grad()
  def decode_latents(self, latents, enable_tiling=True, cpu=True):
    vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
    vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast

    expand_temporal_dim = False
    if len(latents.shape) == 4:
      if isinstance(self.vae, AutoencoderKLCausal3D):
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
      pass
    else:
      raise ValueError(
        f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
      )

    if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
      latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
    else:
      latents = latents / self.vae.config.scaling_factor

    with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
      if enable_tiling:
        self.vae.enable_tiling()
      video = self.vae.decode(latents, return_dict=False)[0]

    if expand_temporal_dim or video.shape[2] == 1:
      video = video.squeeze(2)
    
    video = (video / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 4, 1).contiguous() # [1, 3, 93, 640, 640] -> [1, 93, 640, 640, 3]

    if cpu: 
      return (video*255).to(dtype=torch.uint8).cpu()
    return video


  def make_diffusion_solver_step(self, solver, latents, t, guidance_scale, generator, **diffusion_kwargs):
    do_cfg = guidance_scale>1
    latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
    prompt_embeds = diffusion_kwargs['prompt_embeds']
    prompt_embeds_2 = diffusion_kwargs['prompt_embeds_2']
    prompt_mask = diffusion_kwargs['prompt_mask']
    freqs_cis = diffusion_kwargs['freqs_cis']
    embedded_guidance_scale = diffusion_kwargs['embedded_guidance_scale']
    t_expand = t.repeat(latent_model_input.shape[0]).to(device=diffusion_kwargs['device'])
    guidance_expand = (
      torch.tensor(
        [embedded_guidance_scale] * latent_model_input.shape[0],
        dtype=torch.float32, device=diffusion_kwargs['device'],
      ).to(diffusion_kwargs['target_dtype']) * 1000.0
      if embedded_guidance_scale is not None else None
    )

    with torch.autocast(device_type="cuda", dtype=diffusion_kwargs['target_dtype'], enabled=diffusion_kwargs['autocast_enabled']):


      # print("\n=============================================================")
      # print(1, latent_model_input.shape, t_expand.shape, prompt_embeds.shape, prompt_mask.shape, prompt_embeds_2.shape, freqs_cis[0].shape, freqs_cis[1].shape, guidance_expand.shape)
      
      # print(0,latent_model_input.float().mean().item(), t_expand.float().mean().item(), prompt_embeds.float().mean().item(), prompt_mask.float().mean().item(), prompt_embeds_2.float().mean().item())
      # print(freqs_cis[0].float().mean().item(), freqs_cis[1].float().mean().item(), guidance_expand.float().mean().item())

      noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
        latent_model_input,  # [2, 16, 33, 24, 42]
        t_expand,  # [2]
        text_states=prompt_embeds,  # [2, 256, 4096]
        text_mask=prompt_mask,  # [2, 256]
        text_states_2=prompt_embeds_2,  # [2, 768]
        freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
        freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
        guidance=guidance_expand,
        return_dict=False,
      )[0] # ["x"]

      # print(2, noise_pred.shape)

    assert not torch.any(torch.isnan(noise_pred))

    if do_cfg:
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # print(3, latents.shape, noise_pred.shape)

    # noise_pred  = solver.eps_pred_from("v_prediction", noise_pred, latents)
    new_latents = solver.step(model_output=noise_pred, sample=latents, generator=generator, return_dict=False, prediction_type="flow_matching")

    # print(1,solver.sigmas)
    # print(1,solver.timesteps)

    if not isinstance(new_latents, torch.Tensor): new_latents = new_latents[0]
    return new_latents  


  # ===============================================================================================================
  # UTILS 
  # ===============================================================================================================
  def retrieve_timesteps(self, num_inference_steps=None, timesteps=None, sigmas=None, device=None, **kwargs):
    '''scheduler.set_timesetps(...)'''
    device = self.device if device is None else device
    if timesteps is not None or sigmas is not None:
      self.scheduler.set_timesteps(timesteps=timesteps, device=device, sigmas=sigmas, **kwargs)
      unet_timesteps = self.scheduler.timesteps
      num_inference_steps = len(timesteps)
    else:
      self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
      unet_timesteps = self.scheduler.timesteps
    
    if getattr(self.scheduler, 'unet_timesteps', None) is not None:
      unet_timesteps = self.scheduler.unet_timesteps
    if kwargs.get('unet_timesteps', None) is not None:
      unet_timesteps = kwargs['unet_timesteps']
    
    return unet_timesteps, num_inference_steps


  def encode_prompt(
      self,
      prompt,
      device,
      do_classifier_free_guidance,
      negative_prompt=None,
      prompt_embeds = None,
      attention_mask = None,
      negative_prompt_embeds = None,
      negative_attention_mask = None,
      text_encoder = None,
      data_type = "image",
  ):
      num_videos_per_prompt = 1
      if text_encoder is None:
          text_encoder = self.text_encoder

      if prompt is not None and isinstance(prompt, str):
          batch_size = 1
      elif prompt is not None and isinstance(prompt, list):
          batch_size = len(prompt)
      else:
          batch_size = prompt_embeds.shape[0]

      if prompt_embeds is None:
          text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
          prompt_outputs = text_encoder.encode(
              text_inputs, data_type=data_type, device=device
          )
          prompt_embeds = prompt_outputs.hidden_state
        

          attention_mask = prompt_outputs.attention_mask
          if attention_mask is not None:
              attention_mask = attention_mask.to(device)
              bs_embed, seq_len = attention_mask.shape
              attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
              attention_mask = attention_mask.view(
                  bs_embed * num_videos_per_prompt, seq_len
              )

      if text_encoder is not None:
          prompt_embeds_dtype = text_encoder.dtype
      elif self.transformer is not None:
          prompt_embeds_dtype = self.transformer.dtype
      else:
          prompt_embeds_dtype = prompt_embeds.dtype

      prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

      if prompt_embeds.ndim == 2:
          bs_embed, _ = prompt_embeds.shape
          # duplicate text embeddings for each generation per prompt, using mps friendly method
          prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
          prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
      else:
          bs_embed, seq_len, _ = prompt_embeds.shape
          # duplicate text embeddings for each generation per prompt, using mps friendly method
          prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
          prompt_embeds = prompt_embeds.view(
              bs_embed * num_videos_per_prompt, seq_len, -1
          )

      # get unconditional embeddings for classifier free guidance
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
          uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

          negative_prompt_outputs = text_encoder.encode(
              uncond_input, data_type=data_type, device=device
          )
          negative_prompt_embeds = negative_prompt_outputs.hidden_state

          negative_attention_mask = negative_prompt_outputs.attention_mask
          if negative_attention_mask is not None:
              negative_attention_mask = negative_attention_mask.to(device)
              _, seq_len = negative_attention_mask.shape
              negative_attention_mask = negative_attention_mask.repeat(
                  1, num_videos_per_prompt
              )
              negative_attention_mask = negative_attention_mask.view(
                  batch_size * num_videos_per_prompt, seq_len
              )

      if do_classifier_free_guidance:
          # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
          seq_len = negative_prompt_embeds.shape[1]

          negative_prompt_embeds = negative_prompt_embeds.to(
              dtype=prompt_embeds_dtype, device=device
          )

          if negative_prompt_embeds.ndim == 2:
              negative_prompt_embeds = negative_prompt_embeds.repeat(
                  1, num_videos_per_prompt
              )
              negative_prompt_embeds = negative_prompt_embeds.view(
                  batch_size * num_videos_per_prompt, -1
              )
          else:
              negative_prompt_embeds = negative_prompt_embeds.repeat(
                  1, num_videos_per_prompt, 1
              )
              negative_prompt_embeds = negative_prompt_embeds.view(
                  batch_size * num_videos_per_prompt, seq_len, -1
              )

      return (
          prompt_embeds,
          negative_prompt_embeds,
          attention_mask,
          negative_attention_mask,
      )


  def prepare_latents(
    self,
    batch_size=1,
    num_channels_latents=None,
    height=640,
    width=640,
    video_length=None,
    dtype=None,
    device=None,
    latents=None,
    generator=None,
  ):
    if video_length is None:
      if "884" in self.args.vae: 
        video_length = (93 - 1) // 4 + 1
      elif "888" in self.args.vae: 
        video_length = (93 - 1) // 8 + 1

    num_channels_latents = self.transformer.config.in_channels if num_channels_latents is None else num_channels_latents
    dtype = self.transformer.dtype if dtype is None else dtype
    device = self.device if device is None else device
    
    shape = (
        batch_size,
        num_channels_latents,
        video_length,
        int(height) // self.vae_scale_factor,
        int(width) // self.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )
    else:
        latents = latents.to(device)

    return latents    

  def get_rotary_pos_embed(self, video_length, height, width):
    target_ndim = 3
    ndim = 5 - 2
    if "884" in self.args.vae:
      latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
    elif "888" in self.args.vae:
      latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
    else:
      latents_size = [video_length, height // 8, width // 8]

    if isinstance(self.model.patch_size, int):
      assert all(s % self.model.patch_size == 0 for s in latents_size), (
        f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
        f"but got {latents_size}."
      )
      rope_sizes = [s // self.model.patch_size for s in latents_size]
    elif isinstance(self.model.patch_size, list):
      assert all(
        s % self.model.patch_size[idx] == 0
        for idx, s in enumerate(latents_size)
      ), (
          f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
          f"but got {latents_size}."
      )
      rope_sizes = [
        s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)
      ]

    if len(rope_sizes) != target_ndim:
      rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = self.model.hidden_size // self.model.heads_num
    rope_dim_list = self.model.rope_dim_list
    if rope_dim_list is None:
      rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
      sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
      rope_dim_list,
      rope_sizes,
      theta=self.args.rope_theta,
      use_real=True,
      theta_rescale_factor=1,
    )
    return freqs_cos, freqs_sin

  def show_video(self, video, fps=18):
    import imageio
    import tempfile
    import IPython.display as display
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    assert video.ndim == 5 and video.shape[0] == 1 and video.shape[-1] == 3
    video_np = video.squeeze(0).numpy()  # Shape: (T, H, W, 3)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
      writer = imageio.get_writer(temp_file.name, fps=fps, codec='libx264', quality=5)
      for frame in video_np:
        writer.append_data(frame)
      writer.close()
      return display.display(display.Video(temp_file.name, embed=True))
from loss import *
from lpips_utils import *

from train_utils import *

import time, torch, wandb, tqdm, torch.optim as optim, pickle, gc
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision, open_clip


class Trainer:
  # ==================================================================================================================
  # INIT
  # ==================================================================================================================
  def __init__(self, config):
    seed_everything(42)

    self.config = config
    print(f"init data...", flush=True)
    self._init_data()
    print(f"init metrics...", flush=True)
    self._init_metrics()
    self._init_timestemps()
    self._init_other()

    init_solver = self.config.init_solver if self.config.train_solver else None
    self.pipe = construct_pipeline(self.config.solver, 'CUSTOM', self.config.model, is_train=True, device=self.config.device, init_solver=init_solver)

    gc.collect()
    torch.cuda.empty_cache()
    
    train_params = []

    if self.config.train_timesteps:
      self.ts_logits = torch.nn.Parameter(self.ts_logits, requires_grad=True)
      train_params.append({"params": self.ts_logits, "lr": self.config.lr_ts})
    
    if self.config.train_timesteps_unet:
      self.uts_logits = self.ts_logits.clone().detach()
      self.uts_logits = torch.nn.Parameter(self.uts_logits, requires_grad=True)
      self.unet_timesteps = self.ts_param(self.uts_logits)
      train_params.append({"params": self.uts_logits, "lr": self.config.lr_uts})
    
    if self.config.train_solver:
      self.pipe.scheduler.set_train_solver(timesteps=self.timesteps, device=self.config.device)
      train_params.append({"params": self.pipe.scheduler.train_params, "lr": self.config.lr_solv})

    self.optimizer = optim.Adam(train_params)
    self.global_step = 0

  
  def _init_timestemps(self):
    self.ts_param = TSParam(self.config.ts_param_method)
    timesteps = torch.linspace(0, 999, self.config.nfe + 1).round().flip(0)[:-1]
    if self.config.ts_start_method == 'linear':
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.ts_start_method == 'bad':
      timesteps[1::2] = timesteps[:-1:2] - 10
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.ts_start_method == 'ays':
      timesteps = _get_ays_timesteps_ts(self.config.nfe)
      timesteps = torch.tensor(timesteps)
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.ts_start_method == 'gits':
      from train_utils import _get_gits_timesteps_ts
      timesteps = _get_gits_timesteps_ts(self.config.nfe)
      timesteps = torch.tensor(timesteps)
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    
    else: raise NotImplementedError
    self.timesteps = self.ts_param(self.ts_logits)
    self.unet_timesteps = None
  
  def _init_data(self):
    with open(self.config.p_dataset, 'rb') as f: 
      dataset = pickle.load(f)

    self.train_prompts = dataset['train']['prompts']
    self.val_prompts   = dataset['val']['prompts']

    self.train_latents = dataset['train']['latents'].to('cpu')
    self.val_latents   = dataset['val']['latents'].to('cpu')

    self.train_teacher_latents_out = dataset['train']['latents_out'].to('cpu')
    self.val_teacher_latents_out   = dataset['val']['latents_out'].to('cpu')
  
    if 'imgs' in dataset['train'] and 'imgs' in dataset['val']:
      self.train_teacher_imgs = dataset['train']['imgs'].to('cpu')
      self.val_teacher_imgs   = dataset['val']['imgs'].to('cpu')

  def _init_metrics(self):
    self.train_teacher_features = {}
    self.val_teacher_features   = {}

    # # ADV

    # # INCEPTION
    # if self.config.decode_imgs:
    #   self.inception_model = torchvision.models.inception_v3(pretrained=True)
    #   self.inception_model.aux_logits = False
    #   self.inception_model.fc = torch.nn.Identity() # output: 2048-dim pool3 features
    #   for p in self.inception_model.parameters(): p.requires_grad = False
    #   self.inception_model = self.inception_model.to(self.config.device).eval()
    #   with torch.no_grad():
    #     self.train_teacher_features['INC'] = inc_features_batch(self.train_teacher_imgs, self.inception_model)
    #     self.val_teacher_features['INC']   = inc_features_batch(self.val_teacher_imgs,   self.inception_model)

    # LPIPS
    if self.config.decode_imgs:
      self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to(self.config.device)    
      with torch.no_grad():
        self.train_teacher_features['LPIPS'] = lpips_features_batch(self.train_teacher_imgs, self.lpips_model)
        self.val_teacher_features['LPIPS']   = lpips_features_batch(self.val_teacher_imgs,   self.lpips_model)

    # # CLIP
    # if self.config.decode_imgs:
    #   self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    #   self.clip_model = self.clip_model.to(self.config.device)
    #   with torch.no_grad():
    #     tokenizer = open_clip.get_tokenizer('ViT-B-32')
    #     tokens = tokenizer(self.train_prompts).to(self.config.device)
    #     clip_features = self.clip_model.encode_text(tokens)
    #     clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    #     self.train_teacher_features['CLIP'] = clip_features

    #     tokens = tokenizer(self.val_prompts).to(self.config.device)
    #     clip_features = self.clip_model.encode_text(tokens)
    #     clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    #     self.val_teacher_features['CLIP'] = clip_features

  def _init_other(self):
    # x_t relaxation
    if self.config.relax_radius > 0: 
      self.train_latents_relax = self.train_latents.clone()



  # ==================================================================================================================
  # MAIN TRAIN
  # ==================================================================================================================
  def train(self):
    seed_everything(42)

    self.min_losses = {}
    time_last = time.perf_counter()

    # main loop
    for epoch in tqdm.tqdm(range(self.config.epochs)):
      self.epoch = epoch
      
      train_losses, train_imgs_log = self.train_epoch()
      torch.cuda.synchronize()
      time_train, time_last = time_last, time.perf_counter()
      time_train = time_last - time_train

      val_losses, val_imgs_log = self.validate_epoch()
      torch.cuda.synchronize()
      time_val, time_last = time_last, time.perf_counter()
      time_val = time_last - time_val

      self.log_epoch(train_losses, val_losses, train_imgs_log, val_imgs_log, time_train, time_val)
      
      if self.config.calc_metrics_epoch > 0 and epoch % self.config.calc_metrics_epoch == 0 and epoch:
        self.calc_metrics()
        self.save_ckpt({'FID': self.FID})

        gc.collect()
        torch.cuda.empty_cache()
      
      if self.config.save_solv_params and self.config.train_solver:
        self.save_ckpt(val_losses)

  # ==================================================================================================================
  # TRAIN EPOCH
  # ==================================================================================================================
  def train_epoch(self):
    train_imgs_log = []
    n_imgs_log = int((min(self.config.train_size, 9))**0.5) ** 2
    n_imgs_log = 0 if not self.config.decode_imgs else n_imgs_log
    train_losses = {}

    for batch_start in range(0, self.config.train_size, self.config.batch_size):
      effective_batch_size = min(self.config.batch_size, self.config.train_size - batch_start)
      batch_losses = {}

      latents_batch = self.train_latents[batch_start:batch_start+effective_batch_size].to(self.config.device)
      if self.config.relax_radius > 0:
        latents_batch_relax = self.train_latents_relax[batch_start:batch_start+effective_batch_size].to(self.config.device).clone().detach().requires_grad_(True)

      # mini-batch
      for sub in range(0, effective_batch_size, self.config.mini_batch_size):
        self.update_graph()

        current_size = min(self.config.mini_batch_size, effective_batch_size - sub)
        start_idx = batch_start + sub
        end_idx = start_idx + current_size

        prompts = self.train_prompts[start_idx:end_idx]
        latents = latents_batch_relax[sub:sub+current_size] if self.config.relax_radius > 0 else latents_batch[sub:sub+current_size]
        gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

        if self.config.decode_imgs:
          with torch.set_grad_enabled('LATENT' not in self.config.loss):
            gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
            if len(train_imgs_log) < n_imgs_log: 
              train_imgs_log = train_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
        else:
          gen_imgs = None

        losses = self.get_train_loss(gen_latents, gen_imgs, start_idx, end_idx)
        loss = losses[self.config.loss] / effective_batch_size

        # if self.config.train_adversarial:

        loss.backward()

        for k, v in losses.items():
          train_losses[k] = train_losses.get(k,0) + v.item()
          batch_losses[k] = batch_losses.get(k,0) + v.item()
        
      log_dict = {f'train/batch_{k}': v / effective_batch_size for k, v in batch_losses.items()}
      self.log_clip_grad_step(log_dict)

      # optimization of relaxed x_t
      if self.config.relax_radius > 0: self._optimize_relax(latents_batch_relax, batch_start)

    for k in train_losses:
      train_losses[k] /= self.config.train_size

    return train_losses, train_imgs_log[:n_imgs_log]


  def get_train_loss(self, gen_latents, gen_imgs, start_idx, end_idx):
    '''returns sum-loss over start_idx -> end_idx'''
    losses = {}
    teacher_latents_out = self.train_teacher_latents_out[start_idx:end_idx]

    with torch.set_grad_enabled('LATENT-L1' == self.config.loss):
      losses['LATENT-L1'] = loss_registry['LATENT-L1'](student_latents_out=gen_latents, teacher_latents_out=teacher_latents_out)

    if self.config.decode_imgs:
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx]

      with torch.set_grad_enabled('L1' == self.config.loss):
          losses['L1'] = loss_registry['L1'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs)
      
      with torch.set_grad_enabled('LPIPS' == self.config.loss):
          loss_model = self.lpips_model
          teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features['LPIPS'])
          losses['LPIPS'] = loss_registry['LPIPS'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)

      # with torch.set_grad_enabled('ADV?' == self.config.loss):
      #   ?

      # with torch.set_grad_enabled('CLIP' == self.config.loss):
      #   loss_model = self.clip_model
      #   teacher_features = self.train_teacher_features['CLIP'][start_idx:end_idx]
      #   losses['CLIP'] = loss_registry['CLIP'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)

      # with torch.set_grad_enabled('INC' == self.config.loss):
      #   loss_model = self.inception_model
      #   teacher_features = self.train_teacher_features['INC'][start_idx:end_idx]
      #   losses['INC'] = loss_registry['INC'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)
      
    return losses

  # ==================================================================================================================
  # VALIDATE EPOCH
  # ==================================================================================================================
  @torch.inference_mode()
  def validate_epoch(self):
    self.update_graph()

    val_imgs_log = []
    val_size = min(self.config.val_size, len(self.val_prompts))
    n_imgs_log = int((min(val_size, 9))**0.5) ** 2
    n_imgs_log = 0 if not self.config.decode_imgs else n_imgs_log
    gen_imgs = None
    
    val_losses = {}

    for start_idx in range(0, val_size, self.config.val_batch_size):
      current_size = min(self.config.val_batch_size, val_size - start_idx)
      end_idx = start_idx + current_size

      prompts = self.val_prompts[start_idx:end_idx]
      latents = self.val_latents[start_idx:end_idx].to(self.config.device)
      gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

      if self.config.decode_imgs:
        gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
        if len(val_imgs_log) < n_imgs_log:
          val_imgs_log = val_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
      
      batch_losses = self.get_val_loss(gen_latents, gen_imgs, start_idx, end_idx)

      for k, v in batch_losses.items():
        val_losses[k] = val_losses.get(k, 0.0) + v.item()
      
    val_losses = {k: v / val_size for k, v in val_losses.items()}
    return val_losses, val_imgs_log[:n_imgs_log]
  
  @torch.inference_mode()
  def get_val_loss(self, gen_latents, gen_imgs, start_idx, end_idx):
    '''returns sum-loss over start_idx -> end_idx'''

    losses = {}

    teacher_latents_out = self.val_teacher_latents_out[start_idx:end_idx].to(self.config.device)    
    losses['LATENT-L1'] = loss_registry['LATENT-L1'](student_latents_out=gen_latents, teacher_latents_out=teacher_latents_out)

    if self.config.decode_imgs:
      teacher_imgs = self.val_teacher_imgs[start_idx:end_idx]
      
      loss_model = self.lpips_model
      teacher_features = tuple(i[start_idx:end_idx] for i in self.val_teacher_features['LPIPS'])
      losses['LPIPS'] = loss_registry['LPIPS'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)
      
      losses['L1'] = loss_registry['L1'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs)

      # loss_model = self.clip_model
      # teacher_features = self.val_teacher_features['CLIP'][start_idx:end_idx]
      # losses['CLIP'] = loss_registry['CLIP'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)

      # loss_model = self.inception_model
      # teacher_features = self.val_teacher_features['INC'][start_idx:end_idx]
      # losses['INC'] = loss_registry['INC'](student_imgs=gen_imgs, teacher_imgs=teacher_imgs, teacher_features=teacher_features, loss_model=loss_model)

    return losses


  # ==================================================================================================================
  # LOG
  # ==================================================================================================================
  def log_clip_grad_step(self, log_dict):
    params_to_clip = []
    if self.config.train_timesteps:
      params_to_clip.append(self.ts_logits)
      log_dict['grad_general/ts_grad_norm'] = self.ts_logits.grad.norm(2).item()
      log_dict['grad_general/ts_grad_mean'] = self.ts_logits.grad.mean().item()
      log_dict['grad_general/ts_grad_90%']  = self.ts_logits.grad.abs().quantile(0.9).item()
      log_dict['grad_general/ts_grad_std']  = self.ts_logits.grad.std().item() 
    
    if self.config.train_timesteps_unet:
      params_to_clip.append(self.uts_logits)
      log_dict['grad_general/unet_ts_grad_norm'] = self.uts_logits.grad.norm(2).item()
      log_dict['grad_general/unet_ts_grad_mean'] = self.uts_logits.grad.mean().item()
      log_dict['grad_general/unet_ts_grad_90%']  = self.uts_logits.grad.abs().quantile(0.9).item()
      log_dict['grad_general/unet_ts_grad_std']  = self.uts_logits.grad.std().item()
        
    if self.config.train_solver:
      if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
        params_to_clip.append(self.pipe.scheduler.train_params)
        log_dict['grad_general/solv_norm'] = self.pipe.scheduler.train_params.grad.norm(2).item()
        log_dict['grad_general/solv_mean'] = self.pipe.scheduler.train_params.grad.mean().item()
        log_dict['grad_general/solv_90%']  = self.pipe.scheduler.train_params.grad.abs().quantile(0.9).item()
        log_dict['grad_general/solv_std']  = self.pipe.scheduler.train_params.grad.std().item() 

        for n, p in enumerate(self.pipe.scheduler.train_params.grad):
          log_dict.update({f'solv/grad_mean_nfe[{n}]': p.mean().item()})
          log_dict.update({f'solv/grad_norm_nfe[{n}]': p.norm(2).item()})
          log_dict.update({f'solv/grad_std_nfe[{n}]': p.std().item()})
      else: # it is generator:
        params_to_clip.extend(list(self.pipe.scheduler.train_params))
        
    self.global_step += 1
    wandb.log(log_dict, step=self.global_step)
      
    # clip grads & make optimizer step
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    self.optimizer.step()
    self.optimizer.zero_grad()

  
  def log_epoch(self, train_losses, val_losses, train_imgs_log, val_imgs_log, time_train, time_val):
    log_dict = {'epoch': self.epoch, 'time_train': time_train, 'time_val': time_val}
    log_dict.update({f"train/{k}": v for k, v in train_losses.items()})
    log_dict.update({f"val/{k}": v for k, v in val_losses.items()})

    if self.config.train_timesteps:
      log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(self.timesteps)})
    
    if self.config.train_timesteps_unet:
      log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(self.unet_timesteps)})
    
    if self.config.train_solver and isinstance(self.pipe.scheduler.train_params, torch.Tensor):
      for n, p in enumerate(self.pipe.scheduler.train_params):
        log_dict.update({f'solv/param_mean_nfe[{n}]': p.mean().item()})
        log_dict.update({f'solv/param_std_nfe[{n}]': p.std().item()})
        log_dict.update({f'solv/param_norm_nfe[{n}]': p.norm(2).item()})

    wandb.log(log_dict, step=self.global_step)

    if self.epoch % self.config.img_log_interval == 0 and train_imgs_log and val_imgs_log:
      train_imgs_log = torch.stack(train_imgs_log, dim=0)
      val_imgs_log   = torch.stack(val_imgs_log, dim=0)
      if train_imgs_log.shape[-1] > 224:
        train_imgs_log = torch.nn.functional.interpolate(train_imgs_log, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        val_imgs_log   = torch.nn.functional.interpolate(val_imgs_log,   size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        
        train_imgs_teacher = torch.nn.functional.interpolate(self.train_teacher_imgs[:len(train_imgs_log)], size=(224, 224), mode='bilinear', align_corners=False).squeeze()
        val_imgs_teacher   = torch.nn.functional.interpolate(self.val_teacher_imgs[:len(train_imgs_log)], size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      else:
        train_imgs_teacher = self.train_teacher_imgs[:len(train_imgs_log)]
        val_imgs_teacher   = self.val_teacher_imgs[:len(val_imgs_log)]

      wandb_log_imgs(
        imgs_student=train_imgs_log, 
        imgs_teacher=train_imgs_teacher, 
        key="train",
        global_step=self.global_step,
        )
      wandb_log_imgs(
        imgs_student=val_imgs_log, 
        imgs_teacher=val_imgs_teacher, 
        key="val",
        global_step=self.global_step,
        )

  # ==================================================================================================================
  # METRICS
  # ==================================================================================================================
  @torch.inference_mode()
  def calc_metrics(self):
    init_solver = self.config.init_solver if self.config.train_solver else None
    pipe_metrics = construct_pipeline(
      self.config.solver, 'CUSTOM', self.config.model.replace('_TRAIN', '_BASE'), 
      half=True, device=self.config.device, init_solver = init_solver
    )
    if self.config.train_solver:
      if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
        pipe_metrics.scheduler.train_params = scheduler.train_params.detach().clone().half()
      elif isinstance(self.pipe.scheduler.train_params, list):
        pipe_metrics.scheduler.train_params = []
        for i in range(len(self.pipe.scheduler.train_params)):
          pipe_metrics.scheduler.train_params.append(
            self.pipe.scheduler.train_params[i].detach().clone().half()
          )

    with open('/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/coco30k.pkl', 'rb') as f:
      data = pickle.load(f)
      prompts   = data['anns'][:30000]
      imgs_real = data['imgs'][:30000]
      N = len(prompts)

    imgs_gen = torch.zeros(N, *pipe_metrics.img_dims, dtype=torch.uint8, device='cpu')

    batch_size = 8
    for i in tqdm.tqdm(range(0, N, batch_size), total=N//batch_size, desc='decode...'):
      prompts_batch = prompts[i:i+batch_size]
      generators = [torch.Generator(device='cpu').manual_seed(i+g) for g in range(len(prompts_batch))]
      gen = pipe_metrics(
        prompts_batch, 
        timesteps=self.timesteps,
        unet_timesteps=self.unet_timesteps,
        generator=generators,
        output_type='img'
      ).cpu()
      imgs_gen[i:i+batch_size] = gen
    
    metrics_path = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/lib/metrics'
    if metrics_path not in sys.path: sys.path.insert(0, metrics_path)
    from r_fid  import FID1Metric
    from r_clip import CLIPMetric
    metric_data = dict(
      imgs_gen=imgs_gen, 
      imgs_real=imgs_real, 
      anns=prompts,
      device=self.config.device,
    )
    res_metrics = {
      'metrics/FID':  FID1Metric()(**metric_data),
      'metrics/CLIP': CLIPMetric()(**metric_data)
    }
    self.FID = res_metrics['metrics/FID']
    
    wandb.log(res_metrics, step=self.global_step)
      




  # ==================================================================================================================
  # OTHER
  # ==================================================================================================================  
  def save_ckpt(self, val_losses):
    for loss_name in val_losses:
      if loss_name not in ['LPIPS', 'FID']: continue
      
      curr_loss = val_losses[loss_name]
      prev_best = self.min_losses.get(loss_name, float('inf'))
      if curr_loss < prev_best:
        self.min_losses[loss_name] = curr_loss

        proj_loss_dir = os.path.join(self.config.save_solv_params, wandb.run.project, loss_name)
        if not os.path.exists(proj_loss_dir): os.makedirs(proj_loss_dir)
        with open(os.path.join(proj_loss_dir, wandb.run.name + '.pkl'), 'wb') as f:
          pickle.dump(self.pipe.scheduler.train_params, f)

  
  def update_graph(self):
    if self.config.train_timesteps:
      self.timesteps = self.ts_param(self.ts_logits)
    if self.config.train_timesteps_unet:
      self.unet_timesteps = self.ts_param(self.uts_logits)   
  

  def _optimize_relax(self, latents_batch_relax, batch_start):
    ''' optimize x_T' with projected gradient descent '''
    with torch.no_grad():
      grad_latents_batch_relax = latents_batch_relax.grad
      latents_batch_relax  = latents_batch_relax - self.config.lr_r * grad_latents_batch_relax
      # project back to ball centered at x_T
      for i in range(len(latents_batch_relax)):
        diff = latents_batch[i] - latents_batch_relax[i]
        norm = torch.norm(diff, p=2)
        radius = self.config.relax_radius # 0.001 * d / nfe**2 >> 0.05
        if norm > radius: # project back to the ball's surface
          latents_batch_relax[i] = latents_batch[i] + diff * (radius / norm)
        self.train_latents_relax[batch_start + i] = latents_batch_relax[i]
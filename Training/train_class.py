from loss import *
from lpips_utils import *

from train_utils import *

import time, torch, wandb, tqdm, torch.optim as optim, pickle, gc


class Trainer:
  # ==================================================================================================================
  # INIT
  # ==================================================================================================================
  def __init__(self, config):
    seed_everything(42)

    self.config = config

    init_solver = self.config.solver['init'] if self.config.solver['train'] else None
    self.pipe = construct_pipeline(self.config.solver['solver'], 'CUSTOM', self.config.model, is_train=True, device=self.config.device, init_solver=init_solver)

    print(f"init data...", flush=True)
    self._init_data()
    print(f"init models...", flush=True)
    self._init_models()
    self._init_timestemps()

    gc.collect()
    torch.cuda.empty_cache()
    
    train_params = []

    if self.config.timesteps['train']:
      self.ts_logits = torch.nn.Parameter(self.ts_logits, requires_grad=True)
      train_params.append({"params": self.ts_logits, "lr": self.config.timesteps['lr']})
    
    if self.config.unet_timesteps['train']:
      self.uts_logits = self.ts_logits.clone().detach()
      self.uts_logits = torch.nn.Parameter(self.uts_logits, requires_grad=True)
      self.unet_timesteps = self.ts_param(self.uts_logits)
      train_params.append({"params": self.uts_logits, "lr": self.config.unet_timesteps['lr']})
    
    if self.config.solver['train']:
      self.pipe.scheduler.set_train_solver(timesteps=self.timesteps, device=self.config.device)
      train_params.append({"params": self.pipe.scheduler.train_params, "lr": self.config.solver['lr']})

    self.optimizer = optim.Adam(train_params)
    self.global_step = 0

  
  def _init_timestemps(self):
    self.ts_param = TSParam(self.config.timesteps['param_method'])
    timesteps = torch.linspace(0, 999, self.config.nfe + 1).round().flip(0)[:-1]
    if self.config.timesteps['start_method'] == 'linear':
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.timesteps['start_method'] == 'bad':
      timesteps[1::2] = timesteps[:-1:2] - 10
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.timesteps['start_method'] == 'gits':
      from train_utils import _get_gits_timesteps_ts
      timesteps = _get_gits_timesteps_ts(self.config.nfe)
      timesteps = torch.tensor(timesteps)
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    
    self.timesteps = self.ts_param(self.ts_logits)
    self.unet_timesteps = None
  
  def _init_data(self):
    teacher_filename = f"{self.config.model.split('_')[0]}_{self.config.solver['teacher']}.pkl"
    with open(os.path.join(self.config.dataset['path'], teacher_filename), 'rb') as f: 
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

  def _init_models(self):
    self.train_teacher_features = {}
    self.val_teacher_features   = {}

    # LADV
    if self.config.loss == 'LATENT-ADV':
      from ladv_model.ladv_model import DistAdversarialTraining
      self.ladv_model = DistAdversarialTraining(local_path=self.config.adv['path'], lr=self.config.adv['lr'], device=self.config.device)

    # LADD
    if self.config.loss == 'LATENT-ADD':
      from ladd_model.ladd_model import LADD
      self.ladd_model = LADD(config=self.pipe.scheduler_config, device=self.config.device, freeze=self.config.adv['freeze'], lr=self.config.adv['lr'])

    # LPIPS
    if self.config.img_log_interval > 0:
      from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

      self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to(self.config.device)    
      with torch.no_grad():
        self.train_teacher_features['LPIPS'] = lpips_features_batch(self.train_teacher_imgs, self.lpips_model)
        self.val_teacher_features['LPIPS']   = lpips_features_batch(self.val_teacher_imgs,   self.lpips_model)



  # ==================================================================================================================
  # MAIN TRAIN
  # ==================================================================================================================
  def train(self):
    seed_everything(42)

    self.min_losses = {}
    time_last = time.perf_counter()
    time_metrics = 0

    # main loop
    for epoch in tqdm.tqdm(range(self.config.epochs)):
      self.epoch = epoch
      # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
      train_losses, train_imgs_log = self.train_epoch()
      torch.cuda.synchronize()
      time_train, time_last = time_last, time.perf_counter()
      time_train = time_last - time_train

      val_losses, val_imgs_log = self.validate_epoch()
      torch.cuda.synchronize()
      time_val, time_last = time_last, time.perf_counter()
      time_val = time_last - time_val

      time_train_no_metrics = time_train - time_metrics
      self.log_epoch(train_losses, val_losses, train_imgs_log, val_imgs_log, time_train_no_metrics, time_val)
      
      # cheat with time here:
      time_metrics = time.perf_counter()

      if self.config.calc_metrics_epoch > 0 and epoch % self.config.calc_metrics_epoch == 0 and epoch:
        self.calc_metrics()

        if self.config.solver['train']:
          self.save_ckpt({'FID': self.FID})

        gc.collect()
        torch.cuda.empty_cache()
      
      time_metrics = time.perf_counter() - time_metrics

  # ==================================================================================================================
  # TRAIN EPOCH
  # ==================================================================================================================
  def train_epoch(self):
    train_imgs_log = []
    n_imgs_log = int((min(self.config.dataset['train_size'], 9))**0.5) ** 2
    n_imgs_log = 0 if not self.config.img_log_interval>0 else n_imgs_log
    train_losses = {}

    train_size = self.config.dataset['train_size']
    batch_size = self.config.dataset['batch_size']
    mini_batch_size = self.config.dataset['mini_batch_size']

    for batch_start in range(0, train_size, batch_size):
      effective_batch_size = min(batch_size, train_size - batch_start)
      batch_losses = {}

      latents_batch = self.train_latents[batch_start:batch_start+effective_batch_size].to(self.config.device)
    
      # mini-batch
      for sub in range(0, effective_batch_size, mini_batch_size):
        self.update_graph()

        current_size = min(mini_batch_size, effective_batch_size - sub)
        start_idx = batch_start + sub
        end_idx = start_idx + current_size

        prompts = self.train_prompts[start_idx:end_idx]
        latents = latents_batch[sub:sub+current_size]
        gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

        if self.config.img_log_interval>0:
          with torch.set_grad_enabled('LATENT' not in self.config.loss):
            gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
            if len(train_imgs_log) < n_imgs_log: 
              train_imgs_log = train_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
        else:
          gen_imgs = None

        losses = self.get_train_losses(gen_latents, gen_imgs, start_idx, end_idx)
        (losses[self.config.loss]/effective_batch_size).backward()

        for k, v in losses.items():
          train_losses[k] = train_losses.get(k,0) + v.item()
          batch_losses[k] = batch_losses.get(k,0) + v.item()
      
      log_dict = self.grad_clip()
      self.optimizer.step()
      self.optimizer.zero_grad()

      # DISCRIMINATOR STEP
      if self.config.loss in ['LATENT-ADV', 'LATENT-ADD']:
        if self.config.loss == 'LATENT-ADV':
          adv_model = self.ladv_model
          adv_variant = 'ADV'
        else:
          adv_model = self.ladd_model
          adv_variant = 'ADD'

        # generate different imgs for discriminator (using teachers samples from the end of the dataset)
        ds_sz = self.train_teacher_latents_out.shape[0]
        start_idx = ds_sz - (batch_start + effective_batch_size)
        end_idx   = ds_sz - batch_start

        prompts = self.train_prompts[start_idx:end_idx]
        teacher_latents_out = self.train_teacher_latents_out[start_idx:end_idx].to(self.config.device)
        latents = self.train_latents[start_idx:end_idx].to(self.config.device)

        with torch.no_grad():
          student_latents_out = []
          prompt_embeds = []
          for i in range(effective_batch_size): # lets just take batch=1:
            student_latents_out.append(
              self.pipe(
                prompt=prompts[i:i+1], 
                timesteps=self.timesteps, 
                unet_timesteps=self.unet_timesteps, 
                latents=latents[i:i+1], 
                output_type='latent'
              )
            )
            prompt_embeds.append(self.pipe.prompt_embeds)
          student_latents_out = torch.cat(student_latents_out)
          prompt_embeds = torch.cat(prompt_embeds)

        loss, stats = loss_registry[f'LATENT-{adv_variant}'](
          adv_model, student_latents_out, teacher_latents_out, 
          phase='DIS', 
          scale=self.config.adv['lambda'], 
          gamma=self.config.adv['gamma'], 
          is_train=True,
          prompt_embeds=prompt_embeds
        )
        (loss/effective_batch_size).backward()

        stats[f'LATENT-{adv_variant}-D'] = loss
        for k, v in stats.items():
          train_losses[k] = train_losses.get(k,0) + v.item()
          batch_losses[k] = batch_losses.get(k,0) + v.item()

        log_dict.update(**self.grad_clip_discriminator())
        adv_model.Opt.step()
        adv_model.Opt.zero_grad()
        
      log_dict.update(**{f'train/batch_{k}': v / effective_batch_size for k, v in batch_losses.items()})

      self.global_step += 1
      wandb.log(log_dict, step=self.global_step)

    for k in train_losses:
      train_losses[k] /= train_size

    return train_losses, train_imgs_log[:n_imgs_log]


  def get_train_losses(self, gen_latents, gen_imgs, start_idx, end_idx, **kwargs):
    '''returns sum-loss over start_idx -> end_idx'''
    losses = {}
    teacher_latents_out = self.train_teacher_latents_out[start_idx:end_idx].to(gen_latents.device)

    with torch.set_grad_enabled('LATENT-L1' == self.config.loss):
      losses['LATENT-L1'] = loss_registry['LATENT-L1'](gen_latents, teacher_latents_out)
        
    if self.config.loss == 'LATENT-SL1':
      losses['LATENT-SL1'] = loss_registry['LATENT-SL1'](gen_latents, teacher_latents_out)
    
    if self.config.loss in ['LATENT-ADV', 'LATENT-ADD']:
      if self.config.loss == 'LATENT-ADV':
        adv_model = self.ladv_model
        adv_variant = 'ADV'
      else:
        adv_model = self.ladd_model
        adv_variant = 'ADD'
      prompt_embeds = self.pipe.prompt_embeds
      adv_loss, stats = loss_registry[f'LATENT-{adv_variant}'](
        adv_model, gen_latents, teacher_latents_out, 
        phase='GEN', 
        scale=self.config.adv['lambda'], 
        is_train=True,
        prompt_embeds = prompt_embeds,
        recon_type = self.config.adv['recon_type']
      )
      losses[f'LATENT-{adv_variant}'] = adv_loss
      losses.update(**stats)

    if self.config.img_log_interval>0:
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(gen_latents.device)

      with torch.set_grad_enabled('L1' == self.config.loss):
        losses['L1'] = loss_registry['L1'](gen_imgs, teacher_imgs)
      
      with torch.set_grad_enabled('LPIPS' == self.config.loss):
        teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features['LPIPS'])
        losses['LPIPS'] = loss_registry['LPIPS'](
          self.lpips_model, gen_imgs, teacher_imgs, 
          teacher_features=teacher_features
        )
      
    return losses

  # ==================================================================================================================
  # VALIDATE EPOCH
  # ==================================================================================================================
  @torch.inference_mode()
  def validate_epoch(self):
    self.update_graph()

    val_imgs_log = []
    val_size = min(self.config.dataset['val_size'], len(self.val_prompts))
    n_imgs_log = int((min(val_size, 9))**0.5) ** 2
    n_imgs_log = 0 if not self.config.img_log_interval>0 else n_imgs_log
    gen_imgs = None
    
    val_losses = {}

    for start_idx in range(0, val_size, self.config.dataset['val_batch_size']):
      current_size = min(self.config.dataset['val_batch_size'], val_size - start_idx)
      end_idx = start_idx + current_size

      prompts = self.val_prompts[start_idx:end_idx]
      latents = self.val_latents[start_idx:end_idx].to(self.config.device)
      gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

      if self.config.img_log_interval>0:
        gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
        if len(val_imgs_log) < n_imgs_log:
          val_imgs_log = val_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
      
      batch_losses = self.get_val_losses(gen_latents, gen_imgs, start_idx, end_idx)

      for k, v in batch_losses.items():
        val_losses[k] = val_losses.get(k, 0.0) + v.item()
      
    val_losses = {k: v / val_size for k, v in val_losses.items()}
    return val_losses, val_imgs_log[:n_imgs_log]
  
  @torch.inference_mode()
  def get_val_losses(self, gen_latents, gen_imgs, start_idx, end_idx):
    '''returns sum-loss over start_idx -> end_idx'''
    losses = {}

    teacher_latents_out = self.val_teacher_latents_out[start_idx:end_idx].to(self.config.device)    
    losses['LATENT-L1'] = loss_registry['LATENT-L1'](gen_latents, teacher_latents_out)

    if self.config.img_log_interval>0:
      teacher_imgs = self.val_teacher_imgs[start_idx:end_idx].to(self.config.device)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.val_teacher_features['LPIPS'])
      losses['LPIPS'] = loss_registry['LPIPS'](
        self.lpips_model, gen_imgs, teacher_imgs, 
        teacher_features=teacher_features
      )
      
      losses['L1'] = loss_registry['L1'](gen_imgs, teacher_imgs)
    
    if self.config.loss in ['LATENT-ADV', 'LATENT-ADD']:
      if self.config.loss == 'LATENT-ADV':
        adv_model = self.ladv_model
        adv_variant = 'ADV'
      else:
        adv_model = self.ladd_model
        adv_variant = 'ADD'
      
      adv_variant = self.config.loss.split('-')[-1]
      adv_loss, stats = loss_registry[f'LATENT-{adv_variant}'](
        adv_model, gen_latents, teacher_latents_out, 
        phase='DIS', 
        scale=self.config.adv['lambda'], 
        gamma=self.config.adv['gamma'], 
        is_train=False,
        prompt_embeds = self.pipe.prompt_embeds
      )
      losses[f'LATENT-{adv_variant}-D'] = adv_loss
      losses.update(**stats)

    return losses


  # ==================================================================================================================
  # LOG
  # ==================================================================================================================
  def get_grad_stats(self, grad, suffix):
    return {
      f'{suffix}/norm': grad.norm(2).item(),
      f'{suffix}/mean': grad.mean().item(),
      f'{suffix}/std':  grad.std().item()
    }
  def grad_clip(self):
    log_dict = {}
    params_to_clip = []
    if self.config.timesteps['train']:
      params_to_clip.append(self.ts_logits)
      log_dict.update(**self.get_grad_stats(self.ts_logits.grad, 'grad_ts'))
    
    if self.config.unet_timesteps['train']:
      params_to_clip.append(self.uts_logits)
      log_dict.update(**self.get_grad_stats(self.uts_logits.grad, 'grad_uts'))
        
    if self.config.solver['train']:
      if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
        params_to_clip.append(self.pipe.scheduler.train_params)
        log_dict.update(**self.get_grad_stats(self.pipe.scheduler.train_params.grad, 'grad_solv'))
      else: # it is generator:
        params_to_clip.extend(list(self.pipe.scheduler.train_params))
    
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    return log_dict
  
  def grad_clip_discriminator(self):
    log_dict = {}
    params_to_clip = []
    if self.config.loss == 'LATENT-ADV':
      adv_model = self.ladv_model
      adv_variant = 'grad_ladv'
    else:
      adv_model = self.ladd_model
      adv_variant = 'grad_ladd'
        
    p_train = [p for p in adv_model.discriminator.parameters() if p.requires_grad]
    params_to_clip.extend(p_train)
    grads = torch.cat([p.grad.view(-1) for p in p_train if p.grad is not None])
    log_dict.update(**self.get_grad_stats(grads, adv_variant))

    
    ## may be dont clip it?
    # torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    return log_dict

  def log_epoch(self, train_losses, val_losses, train_imgs_log, val_imgs_log, time_train, time_val):
    log_dict = {'epoch': self.epoch, 'time_train': time_train, 'time_val': time_val}
    log_dict.update({f"train/{k}": v for k, v in train_losses.items()})
    log_dict.update({f"val/{k}": v for k, v in val_losses.items()})

    if self.config.timesteps['train']:
      log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(self.timesteps)})
    
    if self.config.unet_timesteps['train']:
      log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(self.unet_timesteps)})
    
    if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
      log_dict.update({f'solv/params': self.pipe.scheduler.train_params.detach().cpu().numpy().tolist()})
    elif isinstance(self.pipe.scheduler.train_params, list):
      log_dict.update({
        f'solv/params[{n}]': p.detach().cpu().numpy().tolist()
        for n, p in enumerate(self.pipe.scheduler.train_params)
      })

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
    init_solver = self.config.solver['init'] if self.config.solver['train'] else None
    pipe_metrics = construct_pipeline(
      self.config.solver['solver'], 'CUSTOM', self.config.model.replace('_TRAIN', '_BASE'), 
      half=True, device=self.config.device, init_solver = init_solver
    )
    if self.config.solver['train']:
      if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
        pipe_metrics.scheduler.train_params = pipe_metrics.scheduler.train_params.detach().clone().half()
      elif isinstance(self.pipe.scheduler.train_params, list):
        pipe_metrics.scheduler.train_params = []
        for i in range(len(self.pipe.scheduler.train_params)):
          pipe_metrics.scheduler.train_params.append(
            self.pipe.scheduler.train_params[i].detach().clone().half()
          )

    with open('/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/cocold3.pkl', 'rb') as f:
      data = pickle.load(f)
      prompts   = data['anns'][:30000]
      imgs_real = None
      N = len(prompts)

    imgs_gen = torch.zeros(N, *pipe_metrics.img_dims, dtype=torch.uint8, device='cpu')

    print('generating images for FID...', flush=True)
    batch_size = 8
    for i in range(0, N, batch_size):
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
    from r_fid import FIDRef
    metric_data = dict(
      imgs_gen=imgs_gen, 
      imgs_real=imgs_real, 
      anns=prompts,
      device=self.config.device,
    )
    res_metrics = {
      'metrics/FID': FIDRef()(**metric_data),
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

        SAVE_TO = 'DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/SOLV_PARAMS'
        if not os.path.exists(SAVE_TO): os.makedirs(SAVE_TO)
        with open(os.path.join(SAVE_TO, wandb.run.id + '.pkl'), 'wb') as f:
          pickle.dump(self.pipe.scheduler.train_params, f)

  def update_graph(self):
    if self.config.timesteps['train']:
      self.timesteps = self.ts_param(self.ts_logits)
    if self.config.unet_timesteps['train']:
      self.unet_timesteps = self.ts_param(self.uts_logits)   
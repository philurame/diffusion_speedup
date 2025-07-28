from loss import *

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
    self._init_timestemps()
    self._init_models()

    gc.collect()
    torch.cuda.empty_cache()
    
    train_params = []
    n_params_train = 0

    if self.config.timesteps['train']:
      self.ts_logits = torch.nn.Parameter(self.ts_logits.clone().detach(), requires_grad=True)
      train_params.append({"params": self.ts_logits, "lr": self.config.timesteps['lr']})
      n_params_train += self.ts_logits.numel()
        
    if self.config.unet_timesteps['train']:
      self.uts_logits = self.ts_logits.clone().detach()
      self.uts_logits = torch.nn.Parameter(self.uts_logits, requires_grad=True)
      self.unet_timesteps = self.ts_param(self.uts_logits)
      train_params.append({"params": self.uts_logits, "lr": self.config.unet_timesteps['lr']})
      n_params_train += self.uts_logits.numel()
    
    if self.config.solver['train']:
      self.pipe.scheduler.set_train_solver(timesteps=self.timesteps, device=self.config.device)
      train_params.append({"params": self.pipe.scheduler.train_params, "lr": self.config.solver['lr']})
      n_params_train += sum(p.numel() for p in self.pipe.scheduler.train_params)
    
    if self.config.rl['train']:
      init_tensor = -1.*torch.ones(n_params_train, dtype=torch.float32)
      if self.config.solver['train']:
        init_tensor[-sum(p.numel() for p in self.pipe.scheduler.train_params):] = self.config.rl['solver_sigma'] # -2: 0.13; -4: 0.018
      
      self.rl_logits = torch.nn.Parameter(init_tensor, requires_grad=True)

      train_params.append({"params": self.rl_logits, "lr": self.config.rl['lr']})

    self.optimizer = optim.Adam(train_params) # if self.config['optimizer'] == 'SGD': self.optimizer = optim.SGD(train_params, momentum=0.9)
    self.global_step = 0

  
  def _init_timestemps(self):
    self.ts_param = TSParam(self.config.timesteps['param_method'])
    timesteps = torch.linspace(0, 999, self.config.nfe + 1).round().flip(0)[:-1]
    if self.config.timesteps['start_method'] == 'linear':
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.timesteps['start_method'] == 'leading':
      ratio = 1000 // self.config.nfe 
      timesteps = (np.arange(0, self.config.nfe) * ratio).round()[::-1].copy()
      timesteps = torch.tensor(timesteps)
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.timesteps['start_method'] == 'bad':
      timesteps[1::2] = timesteps[:-1:2] - 10
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    elif self.config.timesteps['start_method'] == 'gits':
      from train_utils import get_gits_timesteps_ts
      timesteps = get_gits_timesteps_ts(self.config.nfe)
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

    if self.config.rl['train']:
      self.train_latents = self.train_latents.to(torch.float16)
      self.val_latents   = self.val_latents.to(torch.float16)

    self.train_teacher_latents_out = dataset['train']['latents_out'].to('cpu')
    self.val_teacher_latents_out   = dataset['val']['latents_out'].to('cpu')
  
    if 'imgs' in dataset['train'] and 'imgs' in dataset['val'] and dataset['train']['imgs'] is not None:
      self.train_teacher_imgs = dataset['train']['imgs'].to('cpu')
      self.val_teacher_imgs   = dataset['val']['imgs'].to('cpu')
  
  def _init_models(self):
    if 'IQ' in self.config.loss:
      from pyiqa.archs.musiq_arch import MUSIQ
      cache_dir = '/home/jovyan/.cache/vbench/'
      model_path = f'{cache_dir}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
      self.IQ = MUSIQ(pretrained_model_path=model_path).to(self.config.device)
      self.IQ.training = False
    
    if 'AQ' in self.config.loss:
      import clip
      self.clip_model, preprocess = clip.load('ViT-L/14', device=self.config.device)

      cache_dir = '/home/jovyan/.cache/vbench/aesthetic_model/emb_reader'
      model_path = f"{cache_dir}/sa_0_4_vit_l_14_linear.pth"
      self.AQ = torch.nn.Linear(768, 1).to(self.config.device)
      self.AQ.load_state_dict(torch.load(model_path))
      self.AQ.eval()
    
    if 'MS' in self.config.loss:
      from vbench_utils.ms_utils import MotionSmoothness
      cache_dir = '/home/jovyan/.cache/vbench/'
      ms_config = '/workspace-SR008.fs2/philurame/VMODEL/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'
      ms_ckpt = f'{cache_dir}/amt_model/amt-s.pth'

      self.MS = MotionSmoothness(config=ms_config, ckpt=ms_ckpt, device=self.config.device)



  # ==================================================================================================================
  # MAIN TRAIN
  # ==================================================================================================================
  def train(self):
    seed_everything(42)
    if self.config.calc_metrics_epoch > 0:
      self.calc_metrics()

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

        # if self.config.solver['train']:
        #   self.save_ckpt({'FID': self.FID})

        gc.collect()
        torch.cuda.empty_cache()
      
      time_metrics = time.perf_counter() - time_metrics

  # ==================================================================================================================
  # TRAIN EPOCH
  # ==================================================================================================================
  def update_graph(self):
    if self.config.timesteps['train']:
      self.timesteps = self.ts_param(self.ts_logits)
    if self.config.unet_timesteps['train']:
      self.unet_timesteps = self.ts_param(self.uts_logits)  

  def train_epoch(self):
    train_imgs_log = []
    n_imgs_log = int((min(self.config.dataset['train_size'], 9))**0.5) ** 2
    n_imgs_log = 0 if not self.config.img_log_interval>0 or self.config.rl['train'] else n_imgs_log
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

        if self.config.rl['train']:
          num_samples = self.config.rl['num_samples']

          # RL PARAMS:
          rl_mean_params = []
          if self.config.timesteps['train']:
            rl_mean_params.append(self.ts_logits)
          if self.config.unet_timesteps['train']:
            rl_mean_params.append(self.uts_logits)
          if self.config.solver['train']:
            rl_mean_params += self.pipe.scheduler.train_params
          rl_mean_params = torch.cat(rl_mean_params)

          def not_resample(timesteps_list):
            if timesteps_list is None or timesteps_list[0] is None: return True
            for timesteps in timesteps_list:
              for i in range(len(timesteps)):
                if i>0 and abs(timesteps[i] - timesteps[i-1]) < 0.15:
                  return False
                if timesteps[i]<0. or timesteps[i]>999.5:
                  return False
            return True

          resample_count = 0
          while True:
            self.rl_normal_distr = torch.distributions.Normal(rl_mean_params, torch.exp(self.rl_logits)) 
            self.rl_logits_sample = self.rl_normal_distr.sample(torch.Size([num_samples]))

            idx = 0
            if self.config.timesteps['train']:
              rl_timesteps = [self.ts_param(logits[idx:idx+self.config.nfe]) for logits in self.rl_logits_sample]
              idx += self.config.nfe
            else:
              rl_timesteps = [self.timesteps for _ in range(num_samples)]
            if self.config.unet_timesteps['train']:
              rl_unet_timesteps = [self.ts_param(logits[idx:idx+self.config.nfe]) for logits in self.rl_logits_sample]
              idx += self.config.nfe
            else:
              rl_unet_timesteps = [self.unet_timesteps for _ in range(num_samples)]
            if self.config.solver['train']:
              rl_solver_params = [logits[idx:] for logits in self.rl_logits_sample]
            
            if not_resample(rl_timesteps) and not_resample(rl_unet_timesteps):
              break

            resample_count += 1
            if resample_count > 1000:
              print('resample_count!!!')
              break
          
          # save initial solver_parameters for later:
          orig_solver_params = list(self.pipe.scheduler.train_params) if self.config.solver['train'] else None

          with torch.no_grad():
            gen_latents = []
            for i in range(num_samples):

              if self.config.solver['train']:
                idx=0
                for n in range(len(self.pipe.scheduler.train_params)):
                  self.pipe.scheduler.train_params[n] = rl_solver_params[i][idx:idx+self.pipe.scheduler.train_params[n].numel()]
                  idx += self.pipe.scheduler.train_params[n].numel()
              
              gen_latent = self.pipe(prompt=prompts, timesteps=rl_timesteps[i], unet_timesteps=rl_unet_timesteps[i], latents=latents.to(torch.float16), output_type='latent')
              gen_latents.append(gen_latent)
          
          self.pipe.scheduler.train_params = orig_solver_params
        else:
          gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

        is_decode_latents = self.config.img_log_interval>0 or 'LATENT' not in self.config.loss
        if is_decode_latents:
          is_enable_grad = 'LATENT' not in self.config.loss and not self.config.rl['train']
          with torch.set_grad_enabled(is_enable_grad):
            if isinstance(gen_latents, list): # list of batches of shape [num_samples, batch_size, ...]
              gen_imgs = [self.pipe.decode_latents(gen_latent) * 2 - 1 for gen_latent in gen_latents]
            else:
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
        
      log_dict.update(**{f'train/batch_{k}': v / effective_batch_size for k, v in batch_losses.items()})

      self.global_step += 1
      wandb.log(log_dict, step=self.global_step)

    for k in train_losses:
      train_losses[k] /= train_size

    return train_losses, train_imgs_log[:n_imgs_log]


  def get_train_losses(self, gen_latents, gen_imgs, start_idx, end_idx, **kwargs):
    '''returns sum-loss over start_idx -> end_idx'''
    losses = {}
    teacher_latents_out = self.train_teacher_latents_out[start_idx:end_idx].to(self.config.device)

    if self.config.loss == 'RL-IQ-AQ-MS':
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(self.config.device)
      rl_loss, rl_loss_mean = loss_registry['RL-IQ-AQ-MS'](self.IQ, self.AQ, self.clip_model, self.MS, gen_imgs, coeffs=self.config.other['IQ_AQ_MS'])

    if self.config.loss == 'RL-IQF-AQF-MS':
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(self.config.device)
      rl_loss, rl_loss_mean = loss_registry['RL-IQF-AQF-MS'](self.IQ, self.clip_model, self.MS, gen_imgs, teacher_imgs, coeffs=self.config.other['IQF_AQF_MS'])

    if self.config.loss == 'RL-IQ-AQ-L1':
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(self.config.device)
      rl_loss, rl_loss_mean = loss_registry['RL-IQ-AQ-L1'](self.IQ, self.AQ, self.clip_model, gen_imgs, teacher_imgs, coeffs=self.config.other['IQ_AQ_L1'])

    if self.config.loss == 'RL-IQ':
      rl_loss, rl_loss_mean = loss_registry['RL-IQ'](self.IQ, gen_imgs)
    
    if self.config.loss == 'RL-AQ':
      rl_loss, rl_loss_mean = loss_registry['RL-AQ'](self.AQ, self.clip_model, gen_imgs)

    if self.config.loss == 'RL-LATENT-L1':
      rl_loss, rl_loss_mean = loss_registry['RL-LATENT-L1'](gen_latents, teacher_latents_out)
    
    if self.config.loss == 'RL-L1':
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(self.config.device)
      rl_loss, rl_loss_mean = loss_registry['RL-L1'](gen_imgs, teacher_imgs)

    if 'RL-' in self.config.loss:
      num_samples = self.config.rl['num_samples']
      logprob = self.rl_normal_distr.log_prob(self.rl_logits_sample).sum(dim=1).to(self.config.device) # [num_samples]; TODO: assert shapes
      rl_loss = (rl_loss * logprob[:, None]) * (num_samples) / (num_samples - 1)  # [num_samples, batch_size, ...]

      dims_to_mean = [0] + list(range(2, rl_loss.dim()))
      losses[self.config.loss] = rl_loss.mean(dim=dims_to_mean).sum()
      losses[self.config.loss.replace('RL-','')] = rl_loss_mean
      return losses

    with torch.set_grad_enabled('LATENT-L1' == self.config.loss):
      losses['LATENT-L1'] = loss_registry['LATENT-L1'](gen_latents, teacher_latents_out)
        
    if self.config.loss == 'LATENT-SL1':
      losses['LATENT-SL1'] = loss_registry['LATENT-SL1'](gen_latents, teacher_latents_out)

    if self.config.img_log_interval>0:
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(gen_latents.device)

      with torch.set_grad_enabled('L1' == self.config.loss):
        losses['L1'] = loss_registry['L1'](gen_imgs, teacher_imgs)
  
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

      if self.config.img_log_interval>0 or 'LATENT' not in self.config.loss:
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

    if 'IQ' in self.config.loss:
      losses['IQ'] = loss_registry['IQ'](self.IQ, gen_imgs)
    
    if 'AQ' in self.config.loss:
      losses['AQ'] = loss_registry['AQ'](self.AQ, self.clip_model, gen_imgs)

    teacher_latents_out = self.val_teacher_latents_out[start_idx:end_idx].to(self.config.device)    
    losses['LATENT-L1'] = loss_registry['LATENT-L1'](gen_latents, teacher_latents_out)

    if self.config.img_log_interval>0 or ('L1' in self.config.loss and 'LATENT' not in self.config.loss):
      teacher_imgs = self.val_teacher_imgs[start_idx:end_idx].to(self.config.device)
      losses['L1'] = loss_registry['L1'](gen_imgs, teacher_imgs)

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

    if self.config.rl['train']:
      params_to_clip.append(self.rl_logits)
      log_dict.update(**self.get_grad_stats(self.rl_logits.grad, 'grad_rl'))

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
      else: # it is list:
        params_to_clip.extend(list(self.pipe.scheduler.train_params))
        solv_grad = torch.cat([i.grad for i in self.pipe.scheduler.train_params],  dim=0)
        log_dict.update(**self.get_grad_stats(solv_grad, 'grad_solv'))
    
    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    return log_dict
  

  def log_epoch(self, train_losses, val_losses, train_imgs_log, val_imgs_log, time_train, time_val):
    log_dict = {'epoch': self.epoch, 'time_train': time_train, 'time_val': time_val}
    log_dict.update({f"train/{k}": v for k, v in train_losses.items()})
    log_dict.update({f"val/{k}": v for k, v in val_losses.items()})

    if self.config.timesteps['train']:
      log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(self.timesteps)})

      if self.config.rl['train']:
        log_dict.update({f'logits_rl/logit[{n}]': t.item() for n, t in enumerate(self.rl_logits)})
    
    if self.config.unet_timesteps['train']:
      log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(self.unet_timesteps)})
      
    if self.config.solver['train']:
      if isinstance(self.pipe.scheduler.train_params, torch.Tensor):
        log_dict.update({f'solv/params': self.pipe.scheduler.train_params.detach().cpu().numpy().tolist()})
      elif isinstance(self.pipe.scheduler.train_params, list):
        log_dict.update({
          f'solv/params[{n}]': p.detach().cpu().numpy().tolist()
          for n, p in enumerate(self.pipe.scheduler.train_params)
        })

    wandb.log(log_dict, step=self.global_step)

    if self.config.img_log_interval>0 and self.epoch % self.config.img_log_interval == 0:
      if train_imgs_log:
        train_imgs_log = torch.stack(train_imgs_log, dim=0)[:len(train_imgs_log)]
        train_imgs_teacher = self.train_teacher_imgs[:len(train_imgs_log)]
        wandb_log_imgs(
          imgs_student=train_imgs_log, 
          imgs_teacher=train_imgs_teacher, 
          key="train",
          global_step=self.global_step,
        )
      if val_imgs_log:
        val_imgs_log = torch.stack(val_imgs_log, dim=0)[:len(val_imgs_log)]
        val_imgs_teacher = self.val_teacher_imgs[:len(val_imgs_log)]
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
    if 'SORA' in self.config.model:
      res_metrics = self.calc_VBENCH()
    elif 'SD' in self.config.model:
      res_metrics = self.calc_FID()
    wandb.log(res_metrics, step=self.global_step)
  
  def calc_FID(self):
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
    return res_metrics
  
  def calc_VBENCH(self):
    # GEN VIDEOS
    import imageio, gc, json
    SEED_SHIFT = 0

    prompt_p = '/workspace-SR008.fs2/philurame/VMODEL/huawei_prompts.txt'
    with open(prompt_p, 'r') as f:
      h_prompts = f.readlines()

    _id = wandb.run.id
    video_p = f'/workspace-SR008.fs2/philurame/VMODEL/VIDEOS_TRAIN/{_id}'

    if not os.path.exists(video_p):
      os.makedirs(video_p, exist_ok=True)

    prompt_dict = {}

    for i in tqdm.tqdm(range(len(h_prompts))):
      # if os.path.exists(f'{video_p}/{i}.mp4'): 
      #   prompt_dict[f'{video_p}/{i}.mp4'] = h_prompts[i]
      #   continue

      prompt = h_prompts[i]

      generator = torch.Generator(device='cpu').manual_seed(i+SEED_SHIFT)
      video = self.pipe(prompt=prompt, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, generator=generator, height=640, width=640, output_type='video')
    
      imageio.mimwrite(f'{video_p}/{i}.mp4', 
        video.squeeze(),
        fps=18, 
        quality=10
      )
      prompt_dict[f'{video_p}/{i}.mp4'] = prompt

      torch.cuda.empty_cache()
      gc.collect()
  
    prompts_p = f'/workspace-SR008.fs2/philurame/VMODEL/PROMPTS_TRAIN/{_id}.json'
    with open(prompts_p, 'w') as f:
      json.dump(prompt_dict, f)
    
    # RUN VBENCH
    import subprocess
    shell_path = "/workspace-SR008.fs2/philurame/scripts/help_vbench_train.sh"

    device = str(torch.device(self.config.device).index)
    portn  = str(23031+int(device))

    cmd = [shell_path, device, _id, portn]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # LOG VBENCH
    evaluation_prefix="/workspace-SR008.fs2/philurame/VMODEL/VBench/evaluation_results_train"
    DIMENSIONS=['imaging_quality', 'overall_consistency', 'dynamic_degree', 'subject_consistency', 'aesthetic_quality', 'motion_smoothness']

    vbench_metrics = {}
    for dimension in DIMENSIONS:
      p = f"{evaluation_prefix}/{dimension}/{_id}"
      res_file = [i for i in os.listdir(p) if 'eval_results' in i]
      if len(res_file) == 0:
        print(f"no results for {dimension}!!!!!")
        continue
      res_file = sorted(res_file, key=lambda x: os.path.getmtime(os.path.join(p, x)), reverse=True)[0]
      
      with open(os.path.join(p, res_file), 'r') as f:
        info = json.load(f)
      
      vbench_metrics[dimension] = list(info.values())[0][0]
    
    print('VBENCH DONE!', flush=True)


    import subprocess
    import tempfile
    def fix_mp4_ffmpeg_inplace(path):
      dir_, name = os.path.split(path)
      fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix=name + ".", dir=dir_)
      os.close(fd)

      cmd = [
        "ffmpeg", "-y",
        "-i", path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_path
      ]
      subprocess.run(cmd, check=True)
      os.replace(tmp_path, path)

    for fname in os.listdir(video_p):
      if not fname.lower().endswith(".mp4"): continue
      in_path = os.path.join(video_p, fname)
      fix_mp4_ffmpeg_inplace(in_path)

    return vbench_metrics


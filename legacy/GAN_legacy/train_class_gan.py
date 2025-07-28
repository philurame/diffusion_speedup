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
        init_tensor[-sum(p.numel() for p in self.pipe.scheduler.train_params):] = -8 # 0.0003
      
      self.rl_logits = torch.nn.Parameter(init_tensor, requires_grad=True)

      train_params.append({"params": self.rl_logits, "lr": self.config.rl['lr']})

    if self.config['optimizer'] == 'ADAM':
      self.optimizer = optim.Adam(train_params)
    if self.config['optimizer'] == 'SGD':
      self.optimizer = optim.SGD(train_params, momentum=0.9)
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
              for i in range(1,len(timesteps)):
                if abs(timesteps[i] - timesteps[i-1]) < 0.11:
                  return False
            return True

          resample_count = 0
          while True:
            self.rl_normal_distr = torch.distributions.Normal(rl_mean_params, torch.exp(self.rl_logits)) # self.rl_logits**2
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
                
              gen_latents.append(self.pipe(prompt=prompts, timesteps=rl_timesteps[i], unet_timesteps=rl_unet_timesteps[i], latents=latents.to(torch.float16), output_type='latent'))
          
          self.pipe.scheduler.train_params = orig_solver_params
        else:
          gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')
      
        if self.config.img_log_interval>0 and not self.config.rl['train']:
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

    if self.config.rl['train']: # only LATENT-L1 loss for now
      rl_loss, rl_loss_mean = loss_registry['RL-LATENT-L1'](gen_latents, teacher_latents_out)

      num_samples = self.config.rl['num_samples']
      logprob = self.rl_normal_distr.log_prob(self.rl_logits_sample).sum(dim=1).to(self.config.device) # [num_samples]
      rl_loss = (rl_loss * logprob[:, None]) * (num_samples) / (num_samples - 1)  # [num_samples, batch_size, ...]

      dims_to_mean = [0] + list(range(2, rl_loss.dim()))
      losses['RL-LATENT-L1'] = rl_loss.mean(dim=dims_to_mean).sum()
      losses['LATENT-L1'] = rl_loss_mean

      # assert add

      return losses

    with torch.set_grad_enabled('LATENT-L1' == self.config.loss):
      losses['LATENT-L1'] = loss_registry['LATENT-L1'](gen_latents, teacher_latents_out)
        
    if self.config.loss == 'LATENT-SL1':
      losses['LATENT-SL1'] = loss_registry['LATENT-SL1'](gen_latents, teacher_latents_out)

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
      else: # it is generator:
        params_to_clip.extend(list(self.pipe.scheduler.train_params))
    
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
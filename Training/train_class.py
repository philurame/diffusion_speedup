from train_utils import *
from train_utils import _get_ays_timesteps_ts

import time, torch, wandb, tqdm, torch.optim as optim, pickle
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Trainer:
  # ==================================================================================================================
  # INIT
  # ==================================================================================================================
  def __init__(self, config):
    seed_everything(42)

    self.config = config
    self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to(self.config.device)

    with open(self.config.p_dataset, 'rb') as f:
      dataset = pickle.load(f)
    self.train_prompts = dataset['train']['prompts']
    self.train_latents = dataset['train']['latents'].to('cpu')

    self.train_teacher_latents_out = dataset['train']['latents_out'].to('cpu')
    self.val_prompts = dataset['val']['prompts']
    self.val_latents = dataset['val']['latents'].to('cpu')
    self.val_teacher_latents_out = dataset['val']['latents_out'].to('cpu')
    if 'imgs' in dataset['train'] and 'imgs' in dataset['val']:
      self.train_teacher_imgs = dataset['train']['imgs'].to('cpu')
      self.train_teacher_imgs_224 = torch.nn.functional.interpolate(self.train_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      self.train_teacher_features = get_features(self.train_teacher_imgs_224, self.lpips_model)
      self.val_teacher_imgs = dataset['val']['imgs'].to('cpu')
      self.val_teacher_imgs_224 = torch.nn.functional.interpolate(self.val_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      self.val_teacher_features = get_features(self.val_teacher_imgs_224, self.lpips_model)

    self.pipe = construct_pipeline(self.config.solver, 'CUSTOM', self.config.model, is_train=True, device=self.config.device)

    self.ts_param = TSParam(self.config.ts_param_method)
    self._init_timestemps()
    
    train_params = []

    if self.config.train_timesteps:
      self.ts_logits = torch.nn.Parameter(self.ts_logits, requires_grad=True)
      train_params.append({"params": self.ts_logits, "lr": self.config.lr_ts})
    
    if self.config.train_timesteps_unet:
      self.uts_logits = self.ts_logits.clone().detach()
      self.uts_logits = torch.nn.Parameter(self.uts_logits, requires_grad=True)
      self.unet_timesteps = self.ts_param(self.uts_logits)
      train_params.append({"params": self.uts_logits, "lr": self.config.lr_uts})
    
    if self.pipe.scheduler.is_trainable:
      self.pipe.scheduler.set_train_solver(timesteps=self.timesteps, device=self.config.device)
      train_params.append({"params": self.pipe.scheduler.train_params, "lr": self.config.lr_solv})

    self.optimizer = optim.Adam(train_params)
    self.global_step = 0  

  
  def _init_timestemps(self):
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
    elif self.config.ts_start_method == 'optimal':
      if self.config.nfe == 10:
        timesteps = torch.tensor([999.0, 960.15380859375, 923.48974609375, 885.3140869140625, 844.8024291992188, 784.7818603515625, 700.8524780273438, 547.05419921875, 275.96478271484375, 53.98443603515625])
      if self.config.nfe == 4:
        timesteps = torch.tensor([999.0, 876.153076171875, 692.7985229492188, 404.99066162109375])
      self.ts_logits = self.ts_param.get_logits(timesteps.float())
    else:
      raise NotImplementedError

    self.timesteps = self.ts_param(self.ts_logits)
    self.unet_timesteps = None



  # ==================================================================================================================
  # MAIN TRAIN
  # ==================================================================================================================
  def train(self):
    seed_everything(42)

    self.min_train_loss    = self.min_val_loss    = float('inf')
    self.not_updated_train = self.not_updated_val = 0
    time_last = time.perf_counter()

    # main loop
    for epoch in tqdm.tqdm(range(self.config.epochs)):
      self.epoch = epoch
      
      train_loss, train_imgs_log = self.train_epoch()
      torch.cuda.synchronize()
      time_train, time_last = time_last, time.perf_counter()
      time_train = time_last - time_train

      val_losses, val_imgs_log = self.validate_epoch()
      torch.cuda.synchronize()
      time_val, time_last = time_last, time.perf_counter()
      time_val = time_last - time_val

      self.log_epoch(train_loss, val_losses, train_imgs_log, val_imgs_log, time_train, time_val)

      if self.early_stop(train_loss, val_losses):
        print('Early stopping at epoch', epoch)
        break
    
  
  def early_stop(self, train_loss, val_losses):
    self.not_updated_train += 1
    self.not_updated_val   += 1

    if self.min_train_loss > train_loss:
      self.min_train_loss  = train_loss
      self.not_updated_train = 0
    
    if self.min_val_loss > val_losses[self.config.early_stop_metric]:
      self.min_val_loss  = val_losses[self.config.early_stop_metric]
      self.not_updated_val = 0

      # save best model params:
      if self.config.save_solv_params and self.pipe.scheduler.is_trainable:
        proj_dir = os.path.join(self.config.save_solv_params, wandb.run.project)
        if not os.path.exists(proj_dir): os.makedirs(proj_dir)
        import pickle
        with open(os.path.join(proj_dir, wandb.run.name + '.pkl'), 'wb') as f:
          pickle.dump(self.pipe.scheduler.train_params, f)
    
    if self.not_updated_train >= self.config.early_stop: return True
    if self.not_updated_val   >= self.config.early_stop: return True
    return False



  # ==================================================================================================================
  # TRAIN EPOCH
  # ==================================================================================================================
  def train_epoch(self):
    train_imgs_log = []
    n_imgs_log = min(self.config.train_size, 9)
    n_imgs_log = int((n_imgs_log)**0.5) ** 2
    train_loss = 0

    for batch_start in range(0, self.config.train_size, self.config.batch_size):
      effective_batch_size = min(self.config.batch_size, self.config.train_size - batch_start)
      batch_loss = 0

      # mini-batch
      for sub in range(0, effective_batch_size, self.config.mini_batch_size):
        self.update_graph()

        current_size = min(self.config.mini_batch_size, effective_batch_size - sub)
        start_idx = batch_start + sub
        end_idx = start_idx + current_size

        prompts = self.train_prompts[start_idx:end_idx]
        latents = self.train_latents[start_idx:end_idx].to(self.config.device)

        output_type = 'latent' if self.config.loss.startswith('latent') else 'pt'
        gen_imgs = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type=output_type)
        
        if len(train_imgs_log) < n_imgs_log and not self.config.model.startswith('SORA'):
          log_imgs = gen_imgs[:n_imgs_log]
          if output_type == 'latent':
            with torch.no_grad():
              log_imgs = self.pipe.decode_latents(log_imgs) * 2 - 1
          train_imgs_log = train_imgs_log + [i for i in log_imgs.cpu()]

        raw_loss = self.get_train_loss(gen_imgs, start_idx, end_idx)

        loss = raw_loss / effective_batch_size
        loss.backward()
        train_loss += raw_loss.item()
        batch_loss += raw_loss.item()
      
      batch_loss = batch_loss / effective_batch_size
      log_dict = {f'train/batch_{self.config.loss}' : batch_loss}
      self.log_clip_grad_step(log_dict)
    
    train_loss = train_loss / self.config.train_size

    return train_loss, train_imgs_log[:n_imgs_log]
  
  
  def update_graph(self):
    if self.config.train_timesteps:
      self.timesteps = self.ts_param(self.ts_logits)
    if self.config.train_timesteps_unet:
      self.unet_timesteps = self.ts_param(self.uts_logits)


  def get_train_loss(self, gen_imgs, start_idx, end_idx):
    if self.config.loss.startswith('latent'):
      teacher_latents = self.train_teacher_latents_out[start_idx:end_idx].to(self.config.device)
      if self.config.loss == 'latentl2':
        loss = F.mse_loss(gen_imgs, teacher_latents, reduction='none')
      elif self.config.loss == 'latentl1':
        loss = F.l1_loss(gen_imgs, teacher_latents, reduction='none')
      loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
      return loss

    if self.config.loss == 'lpips':
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features)
      raw_loss = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
      return raw_loss
    
    if self.config.loss.startswith('plpips'):
      l2_mult = self.config.loss == 'plpipsl2'
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx].to(self.config.device)
      plpipsl2 = get_patched_lpips(gen_imgs, teacher_imgs, self.lpips_model, l2_mult=l2_mult)
      return plpipsl2

    if self.config.loss.startswith('l2lpips'):
      integer = self.config.loss.split('l2lpips')[-1]
      integer = int(integer) if integer else 5
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features)
      lpips_loss = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
      l2_loss = F.mse_loss(gen_imgs, self.train_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
      l2_loss = l2_loss.mean(dim=list(range(1,len(l2_loss.shape)))).sum()
      return lpips_loss + l2_loss * integer

    if self.config.loss == 'l1':
      student_imgs = gen_imgs
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx]
      raw_loss = F.l1_loss(student_imgs, teacher_imgs.to(self.config.device), reduction='none')
      raw_loss = raw_loss.mean(dim=list(range(1,len(raw_loss.shape)))).sum()
      return raw_loss

    if self.config.loss.startswith('l2_'):
      resolution = int(self.config.loss.split('_')[-1])
      student_imgs = torch.nn.functional.interpolate(gen_imgs, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
      teacher_imgs = torch.nn.functional.interpolate(self.train_teacher_imgs[start_idx:end_idx], size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
    elif self.config.loss.startswith('crop_'):
      resolution = int(self.config.loss.split('_')[-1])
      student_imgs = gen_imgs[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
    elif self.config.loss == 'l2':
      student_imgs = gen_imgs
      teacher_imgs = self.train_teacher_imgs[start_idx:end_idx]
    
    raw_loss = F.mse_loss(student_imgs, teacher_imgs.to(self.config.device), reduction='none')
    raw_loss = raw_loss.mean(dim=list(range(1,len(raw_loss.shape)))).sum()
    return raw_loss
    

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
        
    if self.pipe.scheduler.is_trainable:
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

    

  # ==================================================================================================================
  # VALIDATE EPOCH
  # ==================================================================================================================
  @torch.inference_mode()
  def validate_epoch(self):
    self.update_graph()

    val_imgs_log = []
    val_size = min(self.config.val_size, len(self.val_prompts))
    n_imgs_log = min(val_size, 9)
    n_imgs_log = int((n_imgs_log)**0.5) ** 2\
    
    n_imgs_log = 0 if self.config.model.startswith('SORA') else n_imgs_log
    
    val_losses = {}

    for start_idx in range(0, val_size, self.config.val_batch_size):
      current_size = min(self.config.val_batch_size, val_size - start_idx)
      end_idx = start_idx + current_size

      prompts = self.val_prompts[start_idx:end_idx]
      latents = self.val_latents[start_idx:end_idx].to(self.config.device)
      gen_latents = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='latent')

      if self.config.model.startswith('SORA'):
        gen_imgs = []
      else:
        gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
        
      if len(val_imgs_log) < n_imgs_log:
        val_imgs_log = val_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
      
      batch_losses = self.get_val_loss(gen_latents, gen_imgs, start_idx, end_idx)

      for k, v in batch_losses.items():
        val_losses[k] = val_losses.get(k, 0.0) + v.item()
      
    val_losses = {k: v / val_size for k, v in val_losses.items()}

    return val_losses, val_imgs_log[:n_imgs_log]
  

  def get_val_loss(self, gen_latents, gen_imgs, start_idx, end_idx):
    losses = {}

    losses['latentl1'] = F.mse_loss(gen_latents, self.val_teacher_latents_out[start_idx:end_idx].to(self.config.device), reduction='none')
    losses['latentl1'] = losses['latentl1'].mean(dim=list(range(1,len(losses['latentl1'].shape)))).sum()

    if self.config.loss.startswith('l2_'):
      resolution = int(self.config.loss.split('_')[-1])
      student_interp = torch.nn.functional.interpolate(gen_imgs, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
      teacher_interp = torch.nn.functional.interpolate(self.train_teacher_imgs[start_idx:end_idx], size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
      losses[self.config.loss] = F.mse_loss(student_interp, teacher_interp.to(self.config.device), reduction='none')
      losses[self.config.loss] = losses[self.config.loss].mean(dim=list(range(1,len(losses[self.config.loss].shape)))).sum()
    
    if self.config.loss.startswith('crop_'):
      resolution = int(self.config.loss.split('_')[-1])
      student_crop = gen_imgs[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
      teacher_crop = self.val_teacher_imgs[start_idx:end_idx, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
      losses[self.config.loss] = F.mse_loss(student_crop, teacher_crop.to(self.config.device), reduction='none')
      losses[self.config.loss] = losses[self.config.loss].mean(dim=list(range(1,len(losses[self.config.loss].shape)))).sum()

    if not self.config.model.startswith('SORA'):
      losses['l2'] = F.mse_loss(gen_imgs, self.val_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
      losses['l2'] = losses['l2'].mean(dim=list(range(1,len(losses['l2'].shape)))).sum()

      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.val_teacher_features)
      losses['lpips'] = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
    
    return losses



  # ==================================================================================================================
  # LOG
  # ==================================================================================================================
  def log_epoch(self, train_loss, val_losses, train_imgs_log, val_imgs_log, time_train, time_val):
    log_dict = {'epoch': self.epoch, 'time_train': time_train, 'time_val': time_val}
    log_dict[f"train/{self.config.loss}"] = train_loss
    log_dict.update({f"val/{k}": v for k, v in val_losses.items()})

    if self.config.train_timesteps:
      log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(self.timesteps)})
    
    if self.config.train_timesteps_unet:
      log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(self.unet_timesteps)})
    
    if self.pipe.scheduler.is_trainable and isinstance(self.pipe.scheduler.train_params, torch.Tensor):
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
        train_imgs_teacher = self.train_teacher_imgs_224[:len(train_imgs_log)]
        val_imgs_teacher   = self.val_teacher_imgs_224[:len(val_imgs_log)]
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

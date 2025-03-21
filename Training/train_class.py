from train_utils import *
from train_utils import _get_ays_timesteps_ts

import torch, wandb, tqdm, torch.optim as optim, pickle
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
  # ==================================================================================================================
  # INIT
  # ==================================================================================================================
  def __init__(self, config):
    seed_everything(42)

    self.config = config
    self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to(config.device)

    with open(config.p_dataset, 'rb') as f:
      dataset = pickle.load(f)
    self.train_prompts = dataset['train']['prompts']
    self.train_latents = dataset['train']['latents']
    self.train_teacher_imgs = dataset['train']['imgs']
    self.val_prompts = dataset['val']['prompts']
    self.val_latents = dataset['val']['latents']
    self.val_teacher_imgs = dataset['val']['imgs']

    self.train_teacher_imgs_224 = torch.nn.functional.interpolate(self.train_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.train_teacher_features = get_features(self.train_teacher_imgs_224, self.lpips_model)
    self.val_teacher_imgs_224 = torch.nn.functional.interpolate(self.val_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.val_teacher_features = get_features(self.val_teacher_imgs_224, self.lpips_model)

    self.pipe = construct_pipeline(config.solver, 'CUSTOM', config.model, is_train=True, device=config.device)
    for param in self.pipe.unet.parameters():
      param.requires_grad = False

    self.ts_param = TSParam(config.ts_param_method)
    self._init_timestemps()
    
    train_params = []
    if config.train_timesteps:
      self.ts_logits = torch.nn.Parameter(self.ts_logits, requires_grad=True)
      train_params.append({"params": self.ts_logits, "lr": config.lr_ts})
    
    if config.train_timesteps_unet:
      self.uts_logits = self.ts_logits.clone().detach()
      self.uts_logits = torch.nn.Parameter(self.uts_logits, requires_grad=True)
      train_params.append({"params": self.uts_logits, "lr": config.lr_uts})
    
    if self.pipe.scheduler.is_trainable:
      self.pipe.scheduler.set_train_solver(self.config.nfe, device=config.device)
      train_params.append({"params": self.pipe.scheduler.train_params, "lr": config.lr_solv})

    self.optimizer = optim.Adam(train_params)
    self.global_step = 0      

  
  def _init_timestemps(self):
    ts_linear = torch.linspace(0, 999, self.config.nfe + 1).round().flip(0)[:-1]
    if self.config.ts_start_method == 'linear':
      self.ts_logits = self.ts_param.get_logits(ts_linear.float())
    elif self.config.ts_start_method == 'bad':
      ts_linear[1::2] = ts_linear[:-1:2] - 10
      self.ts_logits = self.ts_param.get_logits(ts_linear.float())
    elif self.config.ts_start_method == 'ays':
      ts_numpy = _get_ays_timesteps_ts(self.config.nfe)
      ts_linear = torch.tensor(ts_numpy)
      self.ts_logits = self.ts_param.get_logits(ts_linear.float())
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

    # main loop
    for epoch in tqdm.tqdm(range(self.config.epochs)):
      self.epoch = epoch
      train_loss, train_imgs_log = self.train_epoch()
      val_losses, val_imgs_log = self.validate_epoch()
      self.log_epoch(train_loss, val_losses, train_imgs_log, val_imgs_log)

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
        gen_imgs = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='pt')
        
        if len(train_imgs_log) < n_imgs_log:
          train_imgs_log = train_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]

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
    if self.config.loss == 'lpips':
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features)
      raw_loss = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
      return raw_loss

    if self.config.loss == 'l2lpips':
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features)
      lpips_loss = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
      l2_loss = F.mse_loss(gen_imgs, self.train_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
      l2_loss = l2_loss.mean(dim=list(range(1,len(l2_loss.shape)))).sum()
      return lpips_loss + l2_loss * 5

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
    n_imgs_log = int((n_imgs_log)**0.5) ** 2
    
    val_losses = {}

    for start_idx in range(0, val_size, self.config.val_batch_size):
      current_size = min(self.config.val_batch_size, val_size - start_idx)
      end_idx = start_idx + current_size

      prompts = self.val_prompts[start_idx:end_idx]
      latents = self.val_latents[start_idx:end_idx].to(self.config.device)
      gen_imgs = self.pipe(prompt=prompts, timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=latents, output_type='pt')
        
      if len(val_imgs_log) < n_imgs_log:
        val_imgs_log = val_imgs_log + [i for i in gen_imgs[:n_imgs_log].cpu()]
      
      batch_losses = self.get_val_loss(gen_imgs, start_idx, end_idx)

      for k, v in batch_losses.items():
        val_losses[k] = val_losses.get(k, 0.0) + v.item()
      
    val_losses = {k: v / val_size for k, v in val_losses.items()}

    return val_losses, val_imgs_log[:n_imgs_log]
  

  def get_val_loss(self, gen_imgs, start_idx, end_idx):
    losses = {}

    losses['l2'] = F.mse_loss(gen_imgs, self.val_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
    losses['l2'] = losses['l2'].mean(dim=list(range(1,len(losses['l2'].shape)))).sum()

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

    imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    gen_features = get_features(imgs_interp, self.lpips_model)
    teacher_features = tuple(i[start_idx:end_idx] for i in self.val_teacher_features)
    losses['lpips'] = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
    
    return losses



  # ==================================================================================================================
  # LOG
  # ==================================================================================================================
  def log_epoch(self, train_loss, val_losses, train_imgs_log, val_imgs_log):
    log_dict = {'epoch': self.epoch}
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

    if self.epoch % self.config.img_log_interval == 0:
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
      
      # idx = 0 if train_imgs_log.shape[-1] < 224 else 1
      # self.log_difference_heatmaps(train_imgs_log[idx:idx+1], train_imgs_teacher[idx:idx+1])
      # if self.config.log_jacobian:
      #   self.log_saliency_heatmap(idx, interp_n=16)

  

  @torch.inference_mode()
  def log_difference_heatmaps(self, student_img, teacher_img):
    if self.config.loss.startswith('l2') and student_img.shape[-1] > 224:
      resolution = 224 if self.config.loss == 'l2' else int(self.config.loss.split('_')[-1])
      student_img = torch.nn.functional.interpolate(student_img, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
      teacher_img = torch.nn.functional.interpolate(teacher_img, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
    
    if self.config.loss.startswith('crop_'):
      resolution = int(self.config.loss.split('_')[-1])
      student_img = student_img[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
      teacher_img = teacher_img[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]

    diff = (student_img.to(self.config.device) - teacher_img.to(self.config.device)).squeeze().abs().mean(0)
    diff_np = diff.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.heatmap(diff_np, cmap='magma', ax=ax)
    ax.set_title("Absolute Difference")
    ax.set_axis_off()
    wandb.log({f"heatmaps/l1": wandb.Image(fig)}, step=self.global_step)
    plt.close('all')

  def log_saliency_heatmap(self, idx=0, interp_n=16):
    if self.config.loss.startswith('crop_'):
      resolution = int(self.config.loss.split('_')[-1])
    else:
      resolution = None


    orig_data = self.pipe.scheduler.train_params.data.clone()
    grads = torch.zeros((interp_n, interp_n, *self.pipe.scheduler.train_params.shape))

    for i in range(interp_n):
      for j in range(interp_n):
        self.pipe.scheduler.train_params.data.copy_(orig_data)
        self.pipe.scheduler.train_params.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)

        student_img = self.pipe(prompt=self.train_prompts[idx:idx+1], timesteps=self.timesteps, unet_timesteps=self.unet_timesteps, latents=self.train_latents[idx:idx+1].to(self.config.device), output_type='pt')
        teacher_img = self.train_teacher_imgs[idx:idx+1]

        if resolution is not None:
          student_img = student_img[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]
          teacher_img = teacher_img[:, :, resolution//2:-resolution//2, resolution//2:-resolution//2]

        student_img = torch.nn.functional.interpolate(student_img, size=(interp_n, interp_n), mode='bilinear', align_corners=False).squeeze()
        teacher_img = torch.nn.functional.interpolate(teacher_img, size=(interp_n, interp_n), mode='bilinear', align_corners=False).squeeze()
        target_pixel = (student_img[:, i, j] - teacher_img[:, i, j].to(self.pipe.device)).abs().mean()
        target_pixel.backward()

        grads[i,j] = self.pipe.scheduler.train_params.grad.detach().clone()

    self.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
      self.pipe.scheduler.train_params.data.copy_(orig_data)
    

    nfes, nparams = self.pipe.scheduler.train_params.shape
    fig, ax = plt.subplots(nfes, nparams, figsize=(5*nparams, 4*nfes))
    for n in range(nfes):
      for p in range(nparams):
        axi = ax[n, p]
        sns.heatmap(grads[:,:,n,p].detach().cpu().numpy(), cmap='magma', ax=axi)
        axi.set_title(f"nfe[{n}] param[{p}]")

    wandb.log({f"heatmaps/saliency": wandb.Image(fig)}, step=self.global_step)
    plt.close('all')

from train_utils import *

import torch, wandb, tqdm, torch.optim as optim, pickle
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import io
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
    
    with open(config.p_train_prompts, 'rb') as f:
      self.train_prompts = pickle.load(f)
    self.train_latents = torch.load(config.p_train_latents, weights_only=True)[:config.train_size].to(torch.float32)
    self.train_teacher_imgs  = torch.load(config.p_teacher_imgs, weights_only=True)[:config.train_size].to(torch.float32)
    self.train_teacher_imgs_224 = torch.nn.functional.interpolate(self.train_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.train_teacher_features = get_features(self.train_teacher_imgs_224, self.lpips_model)

    with open(config.p_val_dataset, 'rb') as f:
      val_dataset = pickle.load(f)[:config.val_size]
    self.val_prompts = [i[0] for i in val_dataset]
    self.val_latents = torch.stack([i[1] for i in val_dataset]).to(torch.float32)
    self.val_teacher_imgs = torch.stack([i[2] for i in val_dataset]).to(torch.float32)
    self.val_teacher_imgs_224 = torch.nn.functional.interpolate(self.val_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.val_teacher_features = get_features(self.val_teacher_imgs_224, self.lpips_model)

    self.ts_param = TSParam(config.ts_param_method)

    self.pipe = construct_pipeline(config.solver, 'CUSTOM', config.model, is_train=True, device=config.device)
    for param in self.pipe.unet.parameters():
      param.requires_grad = False

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
      self.pipe.scheduler.set_train_solver(self.config.nfe)
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
      train_loss, train_imgs_log = self.train_epoch()
      val_losses, val_imgs_log = self.validate_epoch()
      self.log_epoch(train_loss, val_losses, train_imgs_log, val_imgs_log, epoch)

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

      if self.config.log_debug:
        self.log_difference_heatmaps(gen_imgs[0], start_idx)
    
    train_loss = train_loss / self.config.train_size
    return train_loss, train_imgs_log[:n_imgs_log]
  
  
  def update_graph(self):
    if self.config.train_timesteps:
      self.timesteps = self.ts_param(self.ts_logits)
    if self.config.train_timesteps_unet:
      self.unet_timesteps = self.ts_param(self.uts_logits)


  def get_train_loss(self, gen_imgs, start_idx, end_idx):
    if self.config.loss == 'l2':
      raw_loss = F.mse_loss(gen_imgs, self.train_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
      raw_loss = raw_loss.mean(dim=list(range(1,len(raw_loss.shape)))).sum()

    if self.config.loss == 'l2_224':
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      raw_loss = F.mse_loss(imgs_interp, self.train_teacher_imgs_224[start_idx:end_idx].to(self.config.device), reduction='none')
      raw_loss = raw_loss.mean(dim=list(range(1,len(raw_loss.shape)))).sum()

    elif self.config.loss == 'lpips':
      imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      gen_features = get_features(imgs_interp, self.lpips_model)
      teacher_features = tuple(i[start_idx:end_idx] for i in self.train_teacher_features)
      raw_loss = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
    return raw_loss
  

  def log_clip_grad_step(self, log_dict):
    params_to_clip = []
    if self.config.train_timesteps:
      params_to_clip.append(self.ts_logits)
      log_dict['grad/ts_grad_norm'] = self.ts_logits.grad.norm(2).item()
      log_dict['grad/ts_grad_mean'] = self.ts_logits.grad.mean().item()
      log_dict['grad/ts_grad_90%']  = self.ts_logits.grad.abs().quantile(0.9).item()
      log_dict['grad/ts_grad_std']  = self.ts_logits.grad.std().item() 
      if self.config.log_debug:
        for n, p in enumerate(self.ts_logits):
          log_dict.update({f'grad_debug/t[{n}]': pi.item() for i, pi in enumerate(p.grad)})
    
    if self.config.train_timesteps_unet:
      params_to_clip.append(self.uts_logits)
      log_dict['grad/unet_ts_grad_norm'] = self.uts_logits.grad.norm(2).item()
      log_dict['grad/unet_ts_grad_mean'] = self.uts_logits.grad.mean().item()
      log_dict['grad/unet_ts_grad_90%']  = self.uts_logits.grad.abs().quantile(0.9).item()
      log_dict['grad/unet_ts_grad_std']  = self.uts_logits.grad.std().item()
      if self.config.log_debug:
        for n, p in enumerate(self.uts_logits):
          log_dict.update({f'grad_debug/unet_t[{n}]': pi.item() for i, pi in enumerate(p.grad)})
        
    if self.pipe.scheduler.is_trainable:
      params_to_clip.append(self.pipe.scheduler.train_params)
      log_dict['grad/solv_grad_norm'] = self.pipe.scheduler.train_params.grad.norm(2).item()
      log_dict['grad/solv_grad_mean'] = self.pipe.scheduler.train_params.grad.mean().item()
      log_dict['grad/solv_grad_90%']  = self.pipe.scheduler.train_params.grad.abs().quantile(0.9).item()
      log_dict['grad/solv_grad_std']  = self.pipe.scheduler.train_params.std().item() 
      if self.config.log_debug:
        for n, p in enumerate(self.pipe.scheduler.train_params):
          log_dict.update({f'grad_debug/nfe[{n}]_p[{i}]': pi.item() for i, pi in enumerate(p.grad)})
      
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
    imgs_interp = torch.nn.functional.interpolate(gen_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()

    losses['l2'] = F.mse_loss(gen_imgs, self.val_teacher_imgs[start_idx:end_idx].to(self.config.device), reduction='none')
    losses['l2'] = losses['l2'].mean(dim=list(range(1,len(losses['l2'].shape)))).sum()

    losses['l2_224'] = F.mse_loss(imgs_interp, self.val_teacher_imgs_224[start_idx:end_idx].to(self.config.device), reduction='none')
    losses['l2_224'] = losses['l2_224'].mean(dim=list(range(1,len(losses['l2_224'].shape)))).sum()

    gen_features = get_features(imgs_interp, self.lpips_model)
    teacher_features = tuple(i[start_idx:end_idx] for i in self.val_teacher_features)
    losses['lpips'] = get_lpips(teacher_features, gen_features, self.lpips_model, reduction='sum')
    
    return losses



  # ==================================================================================================================
  # LOG
  # ==================================================================================================================
  def log_epoch(self, train_loss, val_losses, train_imgs_log, val_imgs_log, epoch):
    log_dict = {'epoch': epoch}
    log_dict[f"train/{self.config.loss}"] = train_loss
    log_dict.update({f"val/{k}": v for k, v in val_losses.items()})

    if self.config.train_timesteps:
      log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(self.timesteps)})
    
    if self.config.train_timesteps_unet:
      log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(self.unet_timesteps)})
    
    if self.pipe.scheduler.is_trainable: # shape is (nfe x N)
      for n, p in enumerate(self.pipe.scheduler.train_params):
        log_dict.update({f'solv_params/nfe[{n}]_p[{i}]': pi.item() for i, pi in enumerate(p)})

    wandb.log(log_dict, step=self.global_step)

    if (self.global_step-1) % self.config.img_log_interval == 0:
      train_imgs_log = torch.stack(train_imgs_log, dim=0)
      val_imgs_log   = torch.stack(val_imgs_log, dim=0)
      train_imgs_log = torch.nn.functional.interpolate(train_imgs_log, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      val_imgs_log   = torch.nn.functional.interpolate(val_imgs_log,   size=(224, 224), mode='bilinear', align_corners=False).squeeze()
      wandb_log_imgs(
        imgs_student=train_imgs_log, 
        imgs_teacher=self.train_teacher_imgs_224[:len(train_imgs_log)], 
        key="train",
        global_step=self.global_step,
        )
      wandb_log_imgs(
        imgs_student=val_imgs_log, 
        imgs_teacher=self.val_teacher_imgs_224[:len(val_imgs_log)], 
        key="val",
        global_step=self.global_step,
        )
  
  
  @torch.inference_mode()
  def log_difference_heatmaps(self, gen_img, teacher_idx):
    gen_img = torch.nn.functional.interpolate(gen_img, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    diff = (gen_img - self.train_teacher_imgs_224[teacher_idx].to(self.config.device)).abs()
    if diff.ndim == 3: diff = diff.mean(0)
    diff_np = diff.cpu().numpy()

    plt.figure(figsize=(4,3))
    sns.heatmap(diff_np, cmap='magma')
    plt.title("Absolute Difference")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    wandb.log({f"heatmaps/difference_heatmap": wandb.Image(buf, caption="Absolute Difference Heatmap")}, step=self.global_step)
    plt.close()

  def log_saliency_heatmap(**kwargs):
    """
    TODO: calculates Jacobian of the gen_img with respect to the input solver parameters
    """
    pass


import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from Training.data import TrainDataset
from Training.losses  import loss_registry
from Training.metrics import metric_registry
from Training.models import construct_pipeline, seed_everything
from Training.timesteps import TSModel

from log_utils import log_epoch
from optim_utils import get_optimizer, grad_clip

import time, torch, wandb, tqdm, gc

class Trainer:

  def __init__(self, config):
    seed_everything(42)

    self.config = config

    self.timesteps_model = TSModel(config.nfe, config.timesteps.param_method, config.timesteps.init_method)

    init_solver = config.solver.guide if config.solver.train else None
    self.pipe = construct_pipeline(config.solver.solver, 'CUSTOM', config.model, is_train=True, device=config.device, init_solver=init_solver)
        
    self.train_dataset = TrainDataset('train', config, transfer_device=config.device)
    self.val_dataset   = TrainDataset('val',   config, transfer_device=config.device)

    self.optimizer = get_optimizer(config, self.timesteps_model, self.pipe)
    self.global_step = 1

    self.loss_fn = loss_registry[config.loss.name](device=config.device)

    if config.metric.period > 0:
      self.metric = metric_registry[config.metric.name](run_id = wandb.run.id)
  

  def train(self):
    seed_everything(42)
    time_curr = time.perf_counter()

    # main loop
    for epoch in tqdm.tqdm(range(self.config.epochs)):

      train_loss, train_imgs_log = self.train_epoch()

      torch.cuda.synchronize()
      time_curr, time_last = time.perf_counter(), time_curr 

      val_loss, val_imgs_log = self.validate_epoch()
      
      log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_imgs_log': train_imgs_log, 
        'val_imgs_log': val_imgs_log, 
        'train_imgs_teacher': self.train_dataset[:min(len(train_imgs_log), 9)]['imgs'].float(),
        'val_imgs_teacher': self.val_dataset[:min(len(val_imgs_log), 9)]['imgs'].float(),
        'time_train': time_curr - time_last,
        'global_step': self.global_step,
      }
      log_epoch(self.config, self.timesteps_model, self.pipe, log_data)
  
      if self.config.metric.period > 0 and epoch % self.config.metric.period == 0 and epoch:
        metric_results = self.metric(self.pipe, self.timesteps_model)
        wandb.log(metric_results, step=self.global_step)

        gc.collect()
        torch.cuda.empty_cache()
      

  def train_epoch(self):
    train_loss = 0.

    train_size      = self.config.dataset.train_size
    batch_size      = self.config.dataset.batch_size
    mini_batch_size = self.config.dataset.mini_batch_size

    n_imgs_log = int((min(train_size, 9))**0.5) ** 2 if self.config.img_log_interval > 0 else 0
    train_imgs_log = []

    for batch_start in range(0, train_size, batch_size):
      _batch_size = min(batch_size, train_size - batch_start)
      if _batch_size <= 0: continue
      
      batch_loss = 0.
    
      # mini-batch
      for mini_batch_start in range(0, _batch_size, mini_batch_size):

        _mini_batch_size = min(mini_batch_size, _batch_size - mini_batch_start)
        start_idx = batch_start + mini_batch_start
        end_idx = start_idx + _mini_batch_size

        data = self.train_dataset[start_idx:end_idx]
        timesteps, unet_timesteps = self.timesteps_model()
        gen_latents = self.pipe(
          prompt=data['prompts'], 
          timesteps=timesteps, 
          unet_timesteps=unet_timesteps, 
          latents=data['noise'].to(self.config.device), 
          output_type='latent'
        )

        gen_imgs = None
        if 'LATENT' not in self.config.loss.name:
          gen_imgs = self.pipe.decode_latents(gen_latents)*2-1
          if len(train_imgs_log) < n_imgs_log:
            train_imgs_log.append(gen_imgs.squeeze().cpu().float())

        loss = self.loss_fn(
          gen_latents = gen_latents,
          gen_imgs    = gen_imgs,
          **data
        ).sum()

        (loss/_batch_size).backward()

        batch_loss += loss.item()
        train_loss += loss.item()

      log_dict = grad_clip(self.config, self.timesteps_model, self.pipe)
      log_dict[f'train/batch_{self.config.loss.name}'] = batch_loss / _batch_size
      wandb.log(log_dict, step=self.global_step)
      
      self.optimizer.step()
      self.optimizer.zero_grad()
      self.global_step += 1
      
    return train_loss / train_size, train_imgs_log
  

  @torch.inference_mode()
  def validate_epoch(self):
    seed_everything(42)

    val_imgs_log = []
    val_size = min(self.config.dataset.val_size, len(self.val_dataset))
    n_imgs_log = int((min(val_size, 9))**0.5) ** 2 if self.config.img_log_interval > 0 else 0
    
    val_loss = 0

    for start_idx in range(val_size):
      data = self.val_dataset[start_idx:start_idx+1]
      timesteps, unet_timesteps = self.timesteps_model()
      gen_latents = self.pipe(
        prompt=data['prompts'], 
        timesteps=timesteps, unet_timesteps=unet_timesteps,
        latents=data['noise'], 
        output_type='latent'
      )

      gen_imgs = None
      if self.config.img_log_interval>0:
        gen_imgs = self.pipe.decode_latents(gen_latents) * 2 - 1
        if len(val_imgs_log) < n_imgs_log:
          val_imgs_log.append(gen_imgs.squeeze().cpu().float())

      val_loss += self.loss_fn(
        gen_latents = gen_latents,
        gen_imgs    = gen_imgs,
        **data
      ).sum()
      
    return val_loss/val_size, val_imgs_log

from train_utils import *
from train_class import Trainer

import torch, wandb, tqdm, torch.optim as optim, pickle, sys
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class TrainerCurriculum(Trainer):
  # ==================================================================================================================
  # INIT
  # ==================================================================================================================
  def __init__(self, config):
    seed_everything(42)

    self.config = config
    self.lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to(config.device)

    self.pipe = construct_pipeline(config.solver, 'CUSTOM', 'SDXL_TRAIN', is_train=True, device=config.device)
    
    for param in self.pipe.unet.parameters():
      param.requires_grad = False

    self.ts_param = TSParam('square')
    self._init_timestemps()
    self.timesteps = self.ts_param(self.ts_logits)
    self.unet_timesteps = None
        
    assert self.pipe.scheduler.is_trainable
    self.pipe.scheduler.set_train_solver(self.config.nfe)
    train_params = [{"params": self.pipe.scheduler.train_params, "lr": config.lr_solv}]

    self.optimizer = optim.Adam(train_params)
    self.global_step = 0     

    self.teacher_nfe = self.config.teacher_start
    self.init_dataset()
  
  def init_dataset(self):
    train_size = self.config.train_size
    val_size   = self.config.val_size
    teacher_schedule = self.config.teacher_schedule
    teacher_nfe = self.teacher_nfe
    
    with open(f'/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/TRAIN_DATA/CURRICULUM/DEIS{teacher_schedule}{teacher_nfe}.pkl', 'rb') as f:
      dataset = pickle.load(f)
  
    self.train_prompts = dataset['train']['prompts'][:train_size]
    self.train_latents = dataset['train']['latents'][:train_size].clone()
    self.train_teacher_imgs = dataset['train']['imgs'][:train_size].clone()
    self.val_prompts = dataset['val']['prompts'][:val_size]
    self.val_latents = dataset['val']['latents'][:val_size].clone()
    self.val_teacher_imgs = dataset['val']['imgs'][:val_size].clone()

    self.train_teacher_imgs_224 = torch.nn.functional.interpolate(self.train_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.train_teacher_features = get_features(self.train_teacher_imgs_224, self.lpips_model)
    self.val_teacher_imgs_224 = torch.nn.functional.interpolate(self.val_teacher_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    self.val_teacher_features = get_features(self.val_teacher_imgs_224, self.lpips_model)


  # ==================================================================================================================
  # MAIN TRAIN
  # ==================================================================================================================
  def train(self):
    seed_everything(42)

    wandb.log({'teacher_nfe': self.teacher_nfe}, step=self.global_step)
    optimal_params = {'lpips':float('inf'), 'params': None}

    for epoch in tqdm.tqdm(range(self.config.epochs)):
      self.epoch = epoch
      train_loss, train_imgs_log = self.train_epoch()
      val_losses, val_imgs_log = self.validate_epoch()
      self.log_epoch(train_loss, val_losses, train_imgs_log, val_imgs_log)

      if val_losses['lpips'] < optimal_params['lpips']:
        optimal_params['lpips'] = val_losses['lpips']
        optimal_params['params'] = self.pipe.scheduler.train_params.detach().clone()

      if (epoch+1) % self.config.teacher_epochs == 0:
        self.teacher_nfe += self.config.teacher_step
        self.init_dataset()
        self.pipe.scheduler.train_params.data.copy_(optimal_params['params'])

        for param_group in self.optimizer.param_groups:
          param_group['lr'] *= self.config.lr_decay
        
        wandb.log({'teacher_nfe': self.teacher_nfe}, step=self.global_step)
        optimal_params = {'lpips':float('inf'), 'params': None}
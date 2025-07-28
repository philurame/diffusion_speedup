import torch, wandb
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def log_epoch(config, timesteps_model, pipe, log_data):
  '''train_loss, val_loss, val_imgs_log, time_train, epoch, timesteps_model'''
  log_dict = {'epoch': log_data['epoch'], 'time_train': log_data['time_train']}
  log_dict[f'train/{config.loss.name}'] = log_data['train_loss']
  log_dict[f'val/{config.loss.name}']   = log_data['val_loss']

  if config.timesteps.train:
    log_dict.update({f'timesteps/t[{n}]': t.item() for n, t in enumerate(timesteps_model.timesteps)})
    log_dict.update({f'unet_timesteps/t[{n}]': t.item() for n, t in enumerate(timesteps_model.unet_timesteps)})  

  if config.solver.train:
    if isinstance(pipe.scheduler.train_params, torch.Tensor):
      log_dict['solv/params'] = pipe.scheduler.train_params.detach().cpu().numpy().tolist()
    elif isinstance(pipe.scheduler.train_params, list):
      log_dict['solv/params'] = [p.detach().cpu().numpy().tolist() for p in pipe.scheduler.train_params]

  wandb.log(log_dict, step=log_data['global_step'])

  if config.img_log_interval>0 and log_data['epoch'] % config.img_log_interval == 0:

    is_distill_loss = 'L1' in config.loss.name

    train_imgs_teacher = log_data['train_imgs_teacher'] if is_distill_loss else None
    train_imgs_log     = log_data['train_imgs_log']

    val_imgs_teacher = log_data['val_imgs_teacher'] if is_distill_loss else None
    val_imgs_log     = log_data['val_imgs_log']

    assert train_imgs_log and val_imgs_log, "train_imgs_log and val_imgs_log should not be empty"
    
    # if it is video, take first frame: (t h w c) -> (c h w)
    if train_imgs_log and train_imgs_log[0].squeeze().dim() > 3:
      train_imgs_log = [img[0].permute(2, 0, 1) for img in train_imgs_log]
      val_imgs_log   = [img[0].permute(2, 0, 1) for img in val_imgs_log]
      if is_distill_loss:
        train_imgs_teacher = [img[0].permute(2, 0, 1) for img in train_imgs_teacher]
        val_imgs_teacher   = [img[0].permute(2, 0, 1) for img in val_imgs_teacher]

    wandb_log_imgs(
      imgs_student=train_imgs_log, 
      imgs_teacher=train_imgs_teacher, 
      key="train",
      global_step=log_data['global_step'],
    )
    wandb_log_imgs(
      imgs_student=val_imgs_log, 
      imgs_teacher=val_imgs_teacher, 
      key="val",
      global_step=log_data['global_step'],
    )


# ==================================================================================================================
# WANDB IMG LOG
# ==================================================================================================================
def wandb_log_fig(fig, key, global_step):
  fig.tight_layout()
  wandb.log({key: wandb.Image(fig)}, step=global_step)
  plt.close('all')

def wandb_log_ts(t_steps, global_step=None, key=None, xlabel="NFE", ylabel="TS"):
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.plot(t_steps)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid()
  if global_step is None: return
  wandb_log_fig(fig=fig, key=key, global_step=global_step)

def wandb_log_imgs(imgs_student, imgs_teacher, global_step=None, key=None):
  if isinstance(imgs_student, list):
    imgs_student = torch.stack(imgs_student, dim=0)
  if isinstance(imgs_teacher, list):
    imgs_teacher = torch.stack(imgs_teacher, dim=0)

  if imgs_student.shape[-1] > 224:
    imgs_student = torch.nn.functional.interpolate(imgs_student, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    
    if imgs_teacher is not None:
      imgs_teacher = torch.nn.functional.interpolate(imgs_teacher, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
  
  imgs_student = imgs_student.to(torch.float32)
  if imgs_teacher is not None:
    imgs_teacher = imgs_teacher.to(torch.float32)

  n_cols = 2 if imgs_teacher is not None else 1
  fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
  axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

  vis_grid(imgs_student, ax=axes[0])
  axes[0].axis('off')
  axes[0].set_title("Student")

  if imgs_teacher is not None:
    vis_grid(imgs_teacher, ax=axes[1])
    axes[1].axis('off')
    axes[1].set_title("Teacher")

  if global_step is not None:
    wandb_log_fig(fig=fig, key=key, global_step=global_step)

def vis_grid(imgs_row, ax=None):
  imgs_row = imgs_row.detach().cpu()
  nrow = int(np.around(np.sqrt(imgs_row.shape[0])))
  imgs_grid = make_grid(imgs_row, nrow=nrow).permute(1, 2, 0).numpy()

  imgs_grid = imgs_grid / 2 + 0.5
  imgs_grid = np.clip(imgs_grid, 0, 1)
  if ax is None:
    plt.imshow(imgs_grid)
  else:
    ax.imshow(imgs_grid)
 
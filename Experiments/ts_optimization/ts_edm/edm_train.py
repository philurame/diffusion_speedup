import sys, os, torch, wandb, tqdm, torch.optim as optim, click
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TS_ROOT)

from ts_utils import *


def train(config):
  # ===============================================================================
  lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to('cuda')

  n_imgs = config.train_size+config.val_size
  ddim_imgs = torch.load('/home/mdnikolaev/philurame/DIFFUSION_SPEEDUP/Experiments/ts_optimization/DATA/edm_ddim200_300.pt', weights_only=False)[:n_imgs]
  ddim_imgs_interp = torch.nn.functional.interpolate(ddim_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
  ddim_features = get_features(ddim_imgs_interp, lpips_model)

  pipe = construct_pipeline(config.solver, 'CUSTOM', 'EDM_TRAIN', is_train=True)
  for param in pipe.unet.parameters():
    param.requires_grad = False

  ts_param = TSParam(config.ts_param)

  if config.ts_start == 'linear':
    timesteps_logits = ts_param.get_logits(torch.tensor([999., 899, 799, 699, 599, 500, 400, 300, 200, 100],dtype=torch.float32))
  elif config.ts_start == 'bad':
    timesteps_logits = ts_param.get_logits(torch.tensor([900., 300, 290, 280, 100, 90, 80, 70, 2, 1],dtype=torch.float32))
  timesteps_logits.requires_grad = True

  optimizer = optim.Adam([timesteps_logits], lr=config.learning_rate)
  # ===============================================================================
  
  # TRAIN STARTS ABOUT HERE
  global_step = 0
  validate(config, global_step, pipe, ts_param, timesteps_logits, ddim_features, ddim_imgs, lpips_model)

  for epoch in range(config.epochs):
    imgs_log = torch.zeros(9, 3, 32, 32, dtype=torch.float32, device='cpu')
    for i in tqdm.tqdm(range(0, config.train_size, config.batch_size)):
      global_step += 1

      batch_size = min(config.batch_size, config.train_size-i)
      generators_batch = [torch.Generator(device='cpu').manual_seed(i+j) for j in range(batch_size)]
      ddim_features_batch = {kk: v[i:i+batch_size] for kk, v in ddim_features.items()}

      timesteps = ts_param(timesteps_logits)
      imgs = pipe(timesteps=timesteps, generator=generators_batch)
      imgs_interp = torch.nn.functional.interpolate(imgs, size=(224, 224),mode='bilinear',align_corners=False).squeeze()

      features  = get_features(imgs_interp, lpips_model)
      loss = get_lpips(ddim_features_batch, features, lpips_model)
      loss.backward()

      # log train stats
      wandb.log({
        'grad_norm': timesteps_logits.grad.norm(2).item(),
        'train_loss': loss.item(),
      }, step=global_step)

      torch.nn.utils.clip_grad_norm_([timesteps_logits], max_norm=1.0)
      optimizer.step()
      optimizer.zero_grad()

      imgs_log = torch.cat((imgs_log, imgs[-min(18, imgs.size(0)):].cpu()), dim=0)[-18:]
      
    imgs_student1 = imgs_log[-18:-9]
    imgs_student2 = imgs_log[-9:]
    imgs_teacher1 = ddim_imgs[config.train_size-18:config.train_size-9]
    imgs_teacher2 = ddim_imgs[config.train_size-9:config.train_size]
    wandb_log_imgs(imgs_student=imgs_student1, imgs_teacher=imgs_teacher1, global_step=global_step, key="images_train_1")
    wandb_log_imgs(imgs_student=imgs_student2, imgs_teacher=imgs_teacher2, global_step=global_step, key="images_train_2")
    validate(config, global_step, pipe, ts_param, timesteps_logits, ddim_features, ddim_imgs, lpips_model)

@torch.inference_mode()
def validate(config, global_step, pipe, ts_param, timesteps_logits, ddim_features, ddim_imgs, lpips_model):
  timesteps = ts_param(timesteps_logits)

  avg_loss = 0
  imgs_log = torch.zeros(18, 3, 32, 32, dtype=torch.float32, device='cpu')
  for i in range(config.train_size, config.train_size + config.val_size, config.batch_size):
    batch_size = min(config.batch_size, config.train_size+config.val_size-i)
    generators_batch = [torch.Generator(device='cpu').manual_seed(i+j) for j in range(batch_size)]
    ddim_features_batch = {kk: v[i:i+batch_size] for kk, v in ddim_features.items()}

    imgs = pipe(timesteps=timesteps, generator=generators_batch)
    imgs_interp = torch.nn.functional.interpolate(imgs, size=(224, 224),mode='bilinear',align_corners=False).squeeze()

    features = get_features(imgs_interp, lpips_model)
    loss     = get_lpips(ddim_features_batch, features, lpips_model)
    avg_loss += loss.item()

    imgs_log = torch.cat((imgs_log, imgs[-min(18, imgs.size(0)):].cpu()), dim=0)[-18:]

  wandb.log({"val_loss": avg_loss/config.val_size}, step=global_step)
  wandb_log_ts(timesteps, global_step=global_step, key="timesteps")
  
  imgs_student1 = imgs_log[-18:-9]
  imgs_student2 = imgs_log[-9:]
  imgs_teacher1 = ddim_imgs[-18:-9]
  imgs_teacher2 = ddim_imgs[-9:]
  wandb_log_imgs(imgs_student=imgs_student1, imgs_teacher=imgs_teacher1, global_step=global_step, key="images_val_1")
  wandb_log_imgs(imgs_student=imgs_student2, imgs_teacher=imgs_teacher2, global_step=global_step, key="images_val_2")

@click.command()
@click.option("--solver", type=str)
@click.option("--ts_param", type=str)
@click.option("--ts_start", type=str)
def main(**kwargs):
  with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
    wandb_key = f.read().strip()

  wandb.login(key=wandb_key, relogin=True)
  wandb.init(project="EDM_TRAIN_TIMESTEPS", config={
    "solver": kwargs['solver'],
    "train_size": 250,
    "val_size": 50,
    "batch_size": 25,
    "learning_rate": 0.02,
    "epochs": 40,
    "ts_param": kwargs['ts_param'], # ['cumprod', 'softmax', 'softplus', 'square']
    "ts_start": kwargs['ts_start'],
    "save_code": True,
    "mode": 'offline',
  })
  config = wandb.config

  train(config)

  wandb.finish()


if __name__ == '__main__':
  main()
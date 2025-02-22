import sys, os, torch, wandb, tqdm, torch.optim as optim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

TS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TS_ROOT)

from ts_utils import *


pipe = construct_pipeline('DPMS', 'CUSTOM', 'TRAIN')
for module in pipe.components.values():
  if isinstance(module, torch.nn.Module):
    for param in module.parameters():
      param.requires_grad = False
      
lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to('cuda')

ddim_imgs = torch.load('/home/mdnikolaev/philurame/Experiments/Distil_Diff/ddim_imgs_200.pt', weights_only=False)
ddim_imgs = torch.nn.functional.interpolate(ddim_imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
ddim_features = get_features(ddim_imgs, lpips_model)


# ================================================================================
# TRAIN
# ================================================================================
  

with open(os.path.join(os.path.dirname(ROOT), 'wandb_key.txt'), 'r') as f:
  wandb_key = f.read().strip()

wandb.login(key=wandb_key, relogin=True)
wandb.init(project="SDXL_TRAIN_TIMESTEPS", config={
  "solver": "DPMS",
  "train_size": 25,
  "val_size": 0,
  "batch_size": 25,
  "learning_rate": 1e-2,
  "epochs": 300,
  "ts_param": "probs_cumprod",
  "mode": 'offline',
})
config = wandb.config

ts_param = TSParam(config.ts_param)

init_ts = torch.tensor([999., 875., 724., 538., 330., 153., 52., 14., 3., 0.],dtype=torch.float32)
# init_ts = torch.tensor([999., 899, 799, 699, 599, 500, 400, 300, 200, 100],dtype=torch.float32)

timesteps_logits = ts_param.get_logits(init_ts)
timesteps_logits.requires_grad = True

optimizer = optim.Adam([timesteps_logits], lr=config.learning_rate)

loss_accum = 0.0
n_samples = 0
losses = []

# Training loop: iterating over one epoch with gradient accumulation.
for epoch in range(config.epochs):
  for i in tqdm.tqdm(range(config.train_size)):
    timesteps = ts_param(timesteps_logits)

    seed = torch.Generator(device='cpu').manual_seed(i)
    output = pipe(ANNS[i], timesteps=timesteps, generator=seed, output_type="latent")

    # Decode latent to image, resize it, and normalize.
    image_norm = pipe.vae.decode(output / pipe.vae.config.scaling_factor, return_dict=False)[0] # already [-1, 1]
    image_norm = torch.nn.functional.interpolate(image_norm, size=(224, 224),mode='bilinear',align_corners=False).squeeze()

    features = get_features(image_norm, lpips_model)
    ddim_features_slice = {kk: v[i:i+1] for kk, v in ddim_features.items()}
    item_loss = get_lpips(ddim_features_slice, features, lpips_model)
    # item_loss = (ddim_imgs[i].to('cuda') - image_norm).abs().mean()
    item_loss.backward()

    loss_accum += item_loss.item()
    n_samples += 1

    # Perform an optimizer step after accumulating gradients over "batch_size" examples.
    if (i + 1) % config.batch_size == 0 or i == config.train_size - 1:

      dict_log = {
        "batch_loss": loss_accum / n_samples,
        "grad_norm": timesteps_logits.grad.norm(2).item(),
      }
      print('\nlogits:')
      print(" | ".join([f'{k:.2f} -> {m:.2f}' for k, m in zip(ts_param(init_ts), timesteps_logits.tolist())]))
      print('\nts:')
      print(" | ".join([f'{k:.2f} -> {m:.2f}' for k, m in zip(init_ts.tolist(), timesteps.tolist())]))
      print('\ngrads:')
      print(' | '.join(map(lambda x: f'{x:.2f}', timesteps_logits.grad.tolist())))
      sys.stdout.flush()


      torch.nn.utils.clip_grad_norm_([timesteps_logits], max_norm=1.0)
      optimizer.step()


      wandb.log(dict_log)

      optimizer.zero_grad()
      loss_accum = 0.0
      n_samples = 0
      
wandb.finish()



def wandb_log_fig(fig, key, global_step):
  fig.tight_layout()
  wandb.log({key: wandb.Image(fig)}, step=global_step)
  plt.close('all')

@torch.no_grad()
def wandb_log_timesteps(t_steps, global_step=None, key=None):
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.plot(t_steps)
  ax.set_xlabel("Номер шага")
  ax.set_ylabel("Время")
  ax.grid()
  if global_step is None: return
  wandb_log_fig(fig=fig, key=key, global_step=global_step)

@torch.no_grad()
def log_end_img(x_s, x_t, global_step=None, key=None):
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  vis_grid(x_s, ax=ax[0])
  ax[0].axis('off')
  ax[0].set_title("Student")

  vis_grid(x_t, ax=ax[1])
  ax[1].axis('off')
  ax[1].set_title("Teacher")
  if global_step is None:return

  wandb_log_fig(fig=fig, key=key, global_step=global_step)


from torchvision.utils import make_grid # returns tensor

@torch.no_grad()   
def vis_grid(a, ax=None):
    a = a.detach().cpu()
    if len(a.shape) == 2:
        a = a.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)

    nrow = int(np.around(np.sqrt(a.shape[0])))
    a = make_grid(a, nrow=nrow).permute(1, 2, 0).numpy()
    
    a = a / 2 + 0.5
    a = np.clip(a, 0, 1)
    if ax is None:
        plt.imshow(a)
    else:
        ax.imshow(a)
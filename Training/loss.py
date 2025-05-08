'''
all images should be passed as raw outputs from pipe (with values in [-1,1])
returns raw_loss which is sum-loss over batch
'''

import os, sys, torch, torchvision, torch.nn.functional as F
from lpips_utils import lpips_features, lpips_features_loss

ROOT = os.path.dirname( os.path.dirname(os.path.abspath(__file__)) )
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from lib.registries import ClassRegistry
loss_registry = ClassRegistry()

# ========================================================================================
# LADD
# ========================================================================================
@loss_registry.add_to_registry("LATENT-ADD-G")
def latent_add_g_loss(ladd_model=None, student_latents_out=None, teacher_latents_out=None, prompt_embeds=None, recon_loss_type=None, **kwargs):
  '''MEANS over batch'''
  device = student_latents_out.device
  if recon_loss_type == 'L1':
    kwargs['teacher_imgs'] = kwargs['teacher_imgs'].to(device)
  adv_loss, recon_loss = ladd_model.G_loss(student_latents_out, teacher_latents_out.to(device), prompt_embeds, recon_loss_type=recon_loss_type, **kwargs)
  return adv_loss, recon_loss

@loss_registry.add_to_registry("LATENT-ADD-D")
def latent_add_d_loss(ladd_model=None, student_latents_out=None, teacher_latents_out=None, prompt_embeds=None, **kwargs):
  '''MEANS over batch'''
  device = student_latents_out.device
  loss_gen, loss_real = ladd_model.D_loss(student_latents_out, teacher_latents_out.to(device), prompt_embeds)
  return loss_gen, loss_real

# ========================================================================================
# L1
# ========================================================================================
@loss_registry.add_to_registry("LATENT-L1")
def latent_l1_loss(student_latents_out=None, teacher_latents_out=None, **kwargs):
  device = student_latents_out.device
  loss = F.l1_loss(student_latents_out, teacher_latents_out.to(device), reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss

@loss_registry.add_to_registry("L1")
def l1_loss(student_imgs=None, teacher_imgs=None, **kwargs):
  device = student_imgs.device
  loss = F.l1_loss(student_imgs, teacher_imgs.to(device), reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss 

# ========================================================================================
# LPIPS
# ========================================================================================
@loss_registry.add_to_registry("LPIPS")
def lpips_loss(student_imgs=None, teacher_imgs=None, loss_model=None, teacher_features=None, **kwargs):
  lpips_model = loss_model
  batch_size = kwargs.get('batch_size', 64)

  loss = 0.
  for i in range(0, len(student_imgs), batch_size):
    if teacher_features is not None:
      teacher_features_batch = tuple(feat[i:i+batch_size] for feat in teacher_features)
    else:
      teacher_features_batch = lpips_features(teacher_imgs[i:i+batch_size], lpips_model)
    student_features_batch = lpips_features(student_imgs[i:i+batch_size], lpips_model)
    loss += lpips_features_loss(teacher_features_batch, student_features_batch, lpips_model, reduction='sum')

  return loss 

# ========================================================================================
# CLIP
# ========================================================================================
def clip_normalize(imgs, resize='interpolate'):
  '''imgs values should be in [-1, 1]'''
  imgs = ((imgs+1)/2).clip(0,1)
  if resize == 'interpolate':
    imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
  elif resize == 'crop':
    imgs = torchvision.transforms.functional.center_crop(imgs, (224, 224))
  
  mean = torch.tensor([0.48145466, 0.4578275,  0.40821073],device=imgs.device).view(1, 3, 1, 1)
  std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],device=imgs.device).view(1, 3, 1, 1)
  return (imgs - mean) / std

def clip_features(imgs, clip_model):
  imgs_norm = clip_normalize(imgs)
  features = clip_model.encode_image(imgs_norm)
  features = features / features.norm(dim=-1, keepdim=True)
  return features

@loss_registry.add_to_registry("CLIP")
def clip_loss(student_imgs=None, teacher_imgs=None, loss_model=None, teacher_features=None, **kwargs):
  clip_model = loss_model

  if teacher_features is None:
    with torch.no_grad():
      teacher_features = clip_features(teacher_imgs, clip_model)
  student_features = clip_features(student_imgs, clip_model)

  sim = (student_features * teacher_features).sum(dim=-1)
  loss = (1 - sim).sum()
  return loss

# ========================================================================================
# INCEPTION
# ========================================================================================
def inc_normalize(imgs, resize='interpolate'):
  imgs = ((imgs+1)/2).clip(0,1)
  if resize == 'interpolate':
    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
  elif resize == 'crop':
    imgs = torchvision.transforms.functional.center_crop(imgs, (299, 299))

  mean = torch.tensor([0.485, 0.456, 0.406],device=imgs.device).view(1, 3, 1, 1)
  std  = torch.tensor([0.229, 0.224, 0.225],device=imgs.device).view(1, 3, 1, 1)
  return (imgs - mean) / std

def inc_features(imgs, inception_model):
  imgs_norm = inc_normalize(imgs)
  features = inception_model(imgs_norm)
  return features

def inc_features_batch(imgs, inception_model, bs=100):
  device = next(inception_model.parameters()).device
  feats = []
  for i in range(0, imgs.size(0), bs):
      batch = imgs[i : i + bs].to(device)
      feats.append(inc_features(batch, inception_model))
  return torch.cat(feats, dim=0)

@loss_registry.add_to_registry("INC")
def inc_loss(student_imgs=None, teacher_imgs=None, loss_model=None, teacher_features=None, **kwargs):
  inception_model = loss_model    

  if teacher_features is None:
    with torch.no_grad():
      teacher_features = inc_features(teacher_imgs, inception_model)
  student_features = inc_features(student_imgs, inception_model)

  loss = F.l1_loss(student_features, teacher_features, reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss
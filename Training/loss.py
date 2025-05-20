'''
all images should be passed as raw outputs from pipe (with values in [-1,1])
returns raw_loss which is sum-loss over batch
'''

import os, sys, torch, torch.nn.functional as F
from lpips_utils import lpips_features, lpips_features_loss

ROOT = os.path.dirname( os.path.dirname(os.path.abspath(__file__)) )
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from lib.registries import ClassRegistry
loss_registry = ClassRegistry()

# ========================================================================================
# LADV
# ========================================================================================
@loss_registry.add_to_registry("LATENT-ADV")
def latent_adv_loss(ladv_model, student_latents_out, teacher_latents_out, phase, scale=1, gamma=0.2, is_train=True, recon_type='LL1', **kwargs):
  stats = {}
  if phase == 'GEN':
    conditions = None
    loss, adv_rellogits = ladv_model.AccumulateGeneratorGradients(
      student_latents_out, teacher_latents_out, Conditions=conditions, Scale=scale
    )
    loss = loss.sum()
    stats['ADV-G'] = adv_rellogits[0].sum()

    # add recon_loss:
    if recon_type == 'LL1':
      recon_loss = F.l1_loss(student_latents_out, teacher_latents_out, reduction='none')
    elif recon_type == 'SL1':
      recon_loss = F.smooth_l1_loss(student_latents_out, teacher_latents_out, reduction='none')
    recon_loss = recon_loss.mean(dim=list(range(1,len(recon_loss.shape)))).sum()
    stats['ADV-G_recon'] = recon_loss
    loss = loss + recon_loss

  elif phase == 'DIS':
    conditions = None
    loss, adv_rellogits_penalty = ladv_model.AccumulateDiscriminatorGradients(
      student_latents_out, teacher_latents_out, Conditions=conditions, Gamma=gamma, Scale=scale, is_train=is_train
    )
    loss = loss.sum()
    stats['ADV-D'] = adv_rellogits_penalty[0].sum()
    stats['ADV-D_pen'] = (adv_rellogits_penalty[-1] + adv_rellogits_penalty[-2]).sum()

    rellogits = adv_rellogits_penalty[1]
    stats['ADV-D_accuracy'] = (torch.where(rellogits.abs()<1e-5,torch.zeros_like(rellogits),rellogits.sign())/2+1/2).sum()
    stats['ADV-D_relmean']  = rellogits.sum()
  return loss, stats

# ========================================================================================
# LADD
# ========================================================================================
@loss_registry.add_to_registry("LATENT-ADD")
def latent_add_loss(ladv_model, student_latents_out, teacher_latents_out, phase, scale=1, gamma=0.2, prompt_embeds=None, is_train=True, recon_type='LL1', **kwargs):
  stats = {}
  if phase == 'GEN':
    loss, adv_rellogits = ladv_model.generator_loss(
      student_latents_out, teacher_latents_out, prompt_embeds, scale=scale
    )
    loss = loss.sum()
    stats['ADD-G'] = adv_rellogits[0].sum()

    # add recon_loss:
    if recon_type == 'LL1':
      recon_loss = F.l1_loss(student_latents_out, teacher_latents_out, reduction='none')
    elif recon_type == 'SL1':
      recon_loss = F.smooth_l1_loss(student_latents_out, teacher_latents_out, reduction='none')
    recon_loss = recon_loss.mean(dim=list(range(1,len(recon_loss.shape)))).sum()
    stats['ADD-G_recon'] = recon_loss
    loss = loss + recon_loss

  elif phase == 'DIS':
    loss, adv_rellogits_penalty = ladv_model.discriminator_loss(
      student_latents_out, teacher_latents_out, prompt_embeds, gamma=gamma, scale=scale, is_train=False
    )
    loss = loss.sum()
    stats['ADD-D'] = adv_rellogits_penalty[0].sum()
    stats['ADD-D_pen'] = (adv_rellogits_penalty[-1] + adv_rellogits_penalty[-2]).sum()

    rellogits = adv_rellogits_penalty[1]
    stats['ADD-D_accuracy'] = (torch.where(rellogits.abs()<1e-5,torch.zeros_like(rellogits),rellogits.sign())/2+1/2).sum()
    stats['ADD-D_relmean']  = rellogits.sum()
  return loss, stats

# ========================================================================================
# L1
# ========================================================================================
@loss_registry.add_to_registry("LATENT-SL1")
def latent_sl1_loss(student_latents_out, teacher_latents_out, **kwargs):
  loss = F.smooth_l1_loss(student_latents_out, teacher_latents_out, reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss

@loss_registry.add_to_registry("LATENT-L1")
def latent_l1_loss(student_latents_out, teacher_latents_out, **kwargs):
  loss = F.l1_loss(student_latents_out, teacher_latents_out, reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss

@loss_registry.add_to_registry("L1")
def l1_loss(student_imgs, teacher_imgs, **kwargs):
  loss = F.l1_loss(student_imgs, teacher_imgs, reduction='none')
  loss = loss.mean(dim=list(range(1,len(loss.shape)))).sum()
  return loss

# ========================================================================================
# LPIPS
# ========================================================================================
@loss_registry.add_to_registry("LPIPS")
def lpips_loss(lpips_model, student_imgs, teacher_imgs, teacher_features=None, **kwargs):
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
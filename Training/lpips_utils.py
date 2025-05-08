import torch

def lpips_normalize(imgs):
  imgs_norm = torch.nn.functional.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
  return imgs_norm

def lpips_features(imgs, lpips_net):
  device = next(lpips_net.parameters()).device
  imgs_norm = lpips_normalize(imgs).to(device)
  outs_net = lpips_net.net.forward(lpips_net.scaling_layer(imgs_norm))

  def _normalize_tensor(in_feat, eps=1e-8):
    return in_feat / torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))
  feats = tuple(_normalize_tensor(feat) for feat in outs_net)
  return feats

def lpips_features_batch(imgs, lpips_net, bs=100):
  device = next(lpips_net.parameters()).device
  lpips_net.to('cpu')
  feats = []
  for i in range(0, imgs.size(0), bs):
    batch = imgs[i : i + bs].to('cpu')
    feats.append(lpips_features(batch, lpips_net))

  # [(f1a,f2a),(f1b,f2b),(f1c,f2c)] -> [(f1a,f1b,f1c), (f2a,f2b,f2c)]
  per_layer = list(zip(*feats))
  # print([layer_feats.shape for layer_feats in per_layer], [layer_feats.device for layer_feats in per_layer])
  res = tuple(torch.cat(layer_feats, dim=0) for layer_feats in per_layer)
  lpips_net.to(device)
  return res

def lpips_features_loss(feats1, feats2, lpips_net, reduction='mean'):
  device = next(lpips_net.parameters()).device
  feats1 = [f.to(device) for f in feats1]
  feats2 = [f.to(device) for f in feats2]

  total_loss = []
  for f1, f2, lin in zip(feats1, feats2, lpips_net.lins):
    diff = (f1 - f2)**2
    total_loss.append(lin(diff).mean(dim=[2, 3], keepdim=True).squeeze())
  total_loss = sum(total_loss)

  if reduction == 'none':
    return total_loss
  elif reduction == 'sum':
    return total_loss.sum()
  elif reduction == 'mean':
    return total_loss.sum() / feats1[0].shape[0]
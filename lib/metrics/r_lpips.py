from lib.registries import metric_registry
import torch, tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@metric_registry.add_to_registry('LPIPS')
class LPIPS:
  def __call__(self, **kwargs):
    dataset = kwargs['dataset']
    if dataset != 'COCO':
      return None

    imgs_gen = kwargs['imgs_gen']
    n_imgs = min(10_000, len(imgs_gen))
    batch_size = min(250, n_imgs)
    imgs_gen = imgs_gen[:n_imgs]
    imgs_gen_cropped = torch.nn.functional.interpolate(imgs_gen, size=(224, 224), mode='bilinear', align_corners=False)

    path_ddim_200 = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/imgs_ddim200_224.pt'
    imgs_200  = torch.load(path_ddim_200, weights_only=False, map_location='cpu')[:n_imgs]

    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').net.to('cpu')
    lpips_model.eval()

    diffs_sum = 0
    for i in tqdm.tqdm(range(0, n_imgs, batch_size), desc='LPIPS...'):
      feats_200 = self.get_features(imgs_200[i:i+batch_size], lpips_model)
      feats_gen = self.get_features(imgs_gen_cropped[i:i+batch_size], lpips_model)
      diffs_sum += self.lpips_sum(feats_200, feats_gen, lpips_model)

    return diffs_sum / n_imgs

  @torch.inference_mode()
  def get_features(self, imgs, lpips_net):
    '''imgs of type uint8'''
    device = next(lpips_net.parameters()).device
    imgs_norm = imgs.float() / 127.5 - 1
    imgs_norm = imgs_norm.to(device)
    imgs_norm = lpips_net.scaling_layer(imgs_norm)
    outs = lpips_net.net(imgs_norm)
    feats = tuple(self._normalize_tensor(feat).cpu() for feat in outs)
    return feats
  
  @torch.inference_mode()
  def lpips_sum(self, feats1, feats2, lpips_net):
    device = next(lpips_net.parameters()).device
    feats1 = [f.to(device) for f in feats1]
    feats2 = [f.to(device) for f in feats2]

    total_loss = torch.tensor(0.0, device=device)
    for f1, f2, lin in zip(feats1, feats2, lpips_net.lins):
      diff = (f1 - f2)**2
      total_loss += lin(diff).mean(dim=[2, 3], keepdim=True).sum()
    return total_loss.item()
  
  def _normalize_tensor(self, in_feat, eps=1e-8):
    return in_feat / torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))

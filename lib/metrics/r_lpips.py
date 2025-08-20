from registries import metric_registry
import torch
import lpips
from einops import rearrange

@metric_registry.add_to_registry('LPIPS')
class LPIPS:

  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, net='vgg', batch_size=8, **kwargs):
    '''
    imgs_gen, imgs_real in [0,1], but must be in [-1, 1]
    '''

    self.lpips = lpips.LPIPS(net=net, spatial=False).to(kwargs['device'])
    self.lpips.eval()
    
    for param in self.lpips.parameters():
        param.requires_grad = False

    N = imgs_gen.shape[0]
    outs = []
    for i in range(0, N, batch_size):
      g = imgs_gen[i:i+batch_size].to(kwargs['device'])
      r = imgs_real[i:i+batch_size].to(kwargs['device'])
      g = 2 * g - 1
      r = 2 * r - 1
      out = self.lpips(r, g).squeeze()
      outs.append(out)

    return torch.cat(outs, dim=0).mean().item()
        

@metric_registry.add_to_registry('PLPIPS')
class PatchedLPIPS:
  @torch.inference_mode()
  def __call__(self, imgs_gen, imgs_real, patch_size=224, stride=160, net='vgg', batch_size=8, **kwargs):
        
    self.lpips = lpips.LPIPS(net=net, spatial=False).to(kwargs['device'])
    self.lpips.eval()
        
    for param in self.lpips.parameters():
        param.requires_grad = False

    def extract_patches(images, patch_size, stride):
      patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride) # [B, C, NumPatchesH, NumPatchesW, PatchH, PatchW]
      patches = rearrange(patches, 'b c nh nw h w -> (b nh nw) c h w')
      return patches

    N = imgs_gen.shape[0]
    outs = []
    for i in range(0, N, batch_size):
      g = extract_patches(imgs_gen[i:i+batch_size].to(kwargs['device']), patch_size, stride)
      r = extract_patches(imgs_real[i:i+batch_size].to(kwargs['device']), patch_size, stride)
      g = 2 * g - 1
      r = 2 * r - 1
      out = self.lpips(r, g).squeeze()
      outs.append(out)

    return torch.cat(outs, dim=0).mean().item()
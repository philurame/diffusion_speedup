import torch
import lpips
from einops import rearrange

class PatchedLPIPS:
    def __init__(
        self, 
        patch_size = 224, 
        stride = 160,
        net = 'vgg',
        device = 'cuda:0'
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.device = device
        
        self.lpips = lpips.LPIPS(net=net, spatial=False).to(device)
        self.lpips.eval()
        
        for param in self.lpips.parameters():
            param.requires_grad = False
    
    def extract_patches(self, images):
        patches = images.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # [B, C, NumPatchesH, NumPatchesW, PatchH, PatchW]
        patches = rearrange(patches, 'b c nh nw h w -> (b nh nw) c h w')
        return patches
    
    @torch.no_grad()
    def calculate(self, original, generated, reduction='none'):
        '''
            original, generated must be in [-1, 1] (https://pypi.org/project/lpips/)
        '''

        B = original.shape[0]
        
        original = original.to(self.device)
        generated = generated.to(self.device)
        
        original_patches = self.extract_patches(original)
        generated_patches = self.extract_patches(generated)
        
        original_patches = original_patches.clamp(-1, 1)
        generated_patches = generated_patches.clamp(-1, 1)
        
        lpips_scores = self.lpips(original_patches, generated_patches)
        lpips_scores = lpips_scores.view(B, -1)  # [B * n_patches]
        
        if reduction == 'none':
            return lpips_scores.mean(dim=1)  # [B]
        elif reduction == 'mean':
            return lpips_scores.mean()
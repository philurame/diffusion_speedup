import torch
import lpips
from einops import rearrange

class PatchedLPIPS():
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
    def calculate(self, generated, original=None, prompts=None, reduction='none', **kwargs):
        '''
            original, generated must be in [-1, 1] (https://pypi.org/project/lpips/)
        '''
        if original is None:
            raise ValueError("PatchedLPIPS requires 'original' parameter")
        
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


import torch
from torchvision.transforms.functional import to_pil_image, resize, center_crop, normalize
from hpsv2.img_score import initialize_model, model_dict
from hpsv2.utils import hps_version_map
from huggingface_hub import hf_hub_download
from hpsv2.src.open_clip import get_tokenizer


class HPSMetric():
    def __init__(
        self, 
        hps_version='v2.1',
        device='cuda:0'
    ):
        self.hps_version = hps_version
        self.device = device

        initialize_model()
        self.model = model_dict["model"].to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        ckpt_path  = hf_hub_download("xswu/HPSv2", hps_version_map[self.hps_version])
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["state_dict"])
        
        self.preprocess = model_dict["preprocess_val"]
        self.tokenizer  = get_tokenizer("ViT-H-14")


    @torch.no_grad()
    def calculate(self, generated, original=None, prompts=None, **kwargs):

        generated = (generated + 1) * 0.5 # [-1, 1] -> [0, 1]

        if prompts is None:
            raise ValueError("HPSMetric requires 'prompts' parameter")
        
        images = torch.stack([
            self.preprocess(to_pil_image(img)) 
            for img in generated
        ]).to(self.device)

        text  = self.tokenizer(prompts).to(self.device)
        feats = self.model(images, text)
        score = (feats["image_features"] @ feats["text_features"].T).diagonal()

        return -score   # [B]


class PLPIPS_HPS():
    def __init__(
        self,
        patch_size=224,
        stride=160,
        lpips_net='vgg',
        hps_version='v2.1',
        device='cuda:0'
    ):
        self.device = device
        
        self.patched_lpips = PatchedLPIPS(
            patch_size=patch_size,
            stride=stride,
            net=lpips_net,
            device=device
        )
        
        self.hps_metric = HPSMetric(
            hps_version=hps_version,
            device=device
        )
    
    @torch.no_grad()
    def calculate(self, generated, original=None, prompts=None, reduction='none', **kwargs):

        lpips_score = self.patched_lpips.calculate(
            generated=generated,
            original=original,
            prompts=prompts,
            reduction=reduction,
            **kwargs
        )
        
        hps_score = self.hps_metric.calculate(
            generated=generated,
            original=original,
            prompts=prompts,
            **kwargs
        )

        if reduction == 'mean':
            hps_score = hps_score.mean()
        
        return lpips_score + hps_score
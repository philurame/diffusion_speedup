import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing import List

class PatchedLPIPS:
    def __init__(self, patch_size=224, stride=160, device='cuda'):
        self.stride = stride
        self.patch_size = patch_size
        self.device = device

        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            reduction='mean',
            normalize=False  # inputs are already in [-1, 1]
        ).to(device)


    @staticmethod
    def extract_patches(image, patch_size=224, stride=160):
        patches = []
        h, w = image.shape[-2], image.shape[-1]
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[..., y : y + patch_size, x : x + patch_size]
                patches.append(patch)
        return patches
    
    
    def preprocess(self, data) -> torch.Tensor:
        patches = self.extract_patches(data, self.patch_size, self.stride)
        patches = torch.stack(patches, dim=0)
        patches = patches.reshape(-1, 3, self.patch_size, self.patch_size)
        return patches
    
    
    def calculate(self, original: torch.Tensor, generated_images: List[torch.Tensor]) -> torch.Tensor:
        original = self.preprocess(original).to(self.device)
        generated = self.preprocess(generated_images).to(self.device)
        return self.lpips(generated, original).to(self.device)
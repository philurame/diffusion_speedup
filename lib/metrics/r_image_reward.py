from registries import metric_registry

import torch, tqdm
from torchvision.transforms.functional import to_pil_image
import ImageReward as RM

@metric_registry.add_to_registry('ImageReward')
class IRMetric:
  '''
  to_pil_image is not effective but what to do
  '''
  model_name_or_path="ImageReward-v1.0"
  @torch.inference_mode()
  def __call__(self, imgs_gen, prompts, device=None, verbose=False, **kwargs):
    device = device or imgs_gen.device
    model = RM.load(self.model_name_or_path, device=device)
    model.eval()
    
    total = 0.0
    for i in tqdm.tqdm(range(0, len(imgs_gen)), disable=not verbose):
      total += model.score(prompts[i], to_pil_image(imgs_gen[i].to(device)))
    return total / len(imgs_gen)

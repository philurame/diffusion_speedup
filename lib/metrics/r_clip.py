from registries import metric_registry

from torchmetrics.multimodal.clip_score import CLIPScore
import torch, tqdm

@metric_registry.add_to_registry('CLIP')
class CLIPMetric:
  model_name_or_path="openai/clip-vit-large-patch14"
  @torch.inference_mode()
  def __call__(self, imgs_gen, prompts, batch_size=64, device=None, verbose=False, **kwargs):
    torch.manual_seed(0)
    device = device or imgs_gen.device
    clip_model = CLIPScore(model_name_or_path=self.model_name_or_path).to(device)
    clip_model.eval()
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), disable=not verbose):
      imgs = imgs_gen[i:i+batch_size].to(device)
      clip_model.update(imgs, prompts[i:i+len(imgs)])
    return clip_model.compute().item()
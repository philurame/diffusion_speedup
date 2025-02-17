from lib.registries import metric_registry
import torch, tqdm
from torchmetrics.multimodal.clip_score import CLIPScore


@metric_registry.add_to_registry('CLIP')
class CLIPMetric:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    anns = kwargs['anns']
    imgs_gen = kwargs['imgs_gen']

    clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to('cuda')
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), desc='CLIP...'):
      clip_model.update(imgs_gen[i:i+batch_size].to('cuda'), anns[i:i+batch_size])
    return clip_model.compute().item()
from lib.registries import metric_registry
import torch, tqdm
from torchmetrics.image.inception import InceptionScore


@metric_registry.add_to_registry('IS')
class IS:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    imgs_gen = kwargs['imgs_gen']

    is_model = InceptionScore(normalize=False).to('cuda')
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), desc='IS...'):
      is_model.update(imgs_gen[i:i+batch_size].to('cuda'))

    return is_model.compute()[0].item()
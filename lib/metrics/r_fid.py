from lib.registries import metric_registry
import torch, tqdm
from torchmetrics.image.fid import FrechetInceptionDistance


@metric_registry.add_to_registry('FID')
class FIDMetric:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    imgs_gen  = kwargs['imgs_gen']
    imgs_real = kwargs['imgs_real']

    fid_model = FrechetInceptionDistance(feature=2048, normalize=False).to('cuda')
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), desc='FID...'):
      fid_model.update(imgs_real[i:i+batch_size].to('cuda'), real=True)
      fid_model.update(imgs_gen[i:i+batch_size].to('cuda'), real=False)

    return fid_model.compute().item()
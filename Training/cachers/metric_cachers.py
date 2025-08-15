import torch, tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

class FIDMetric:
  @torch.inference_mode()
  def __call__(self, **kwargs):
    batch_size = kwargs.get('batch_size', 512)
    imgs_gen  = kwargs['imgs_gen']
    imgs_real = kwargs['imgs_real']
    device = kwargs['device']

    fid_model = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for i in tqdm.tqdm(range(0, imgs_gen.shape[0], batch_size), desc='FID...'):
      fid_model.update(imgs_real[i:i+batch_size].to(device), real=True)
      fid_model.update(imgs_gen[i:i+batch_size].to(device), real=False)

    return fid_model.compute().item()
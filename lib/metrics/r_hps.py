from registries import metric_registry

import torch, tqdm
from torchvision.transforms.functional import to_pil_image, resize, center_crop, normalize
from hpsv2.img_score import initialize_model, model_dict
from hpsv2.utils import hps_version_map
from huggingface_hub import hf_hub_download
from hpsv2.src.open_clip import get_tokenizer

@metric_registry.add_to_registry('HPS')
class HPSMetric:
  hps_version = 'v2.1'
  @torch.inference_mode()
  def __call__(self, imgs_gen, prompts, batch_size=64, device=None, verbose=False, **kwargs):
    device = device or imgs_gen.device

    if isinstance(device, torch.device) and device.type == 'cuda':
      torch.cuda.set_device(device)
    elif isinstance(device, str) and 'cuda' in device:
      device_id = int(device.split(':')[-1]) if ':' in device else 0
      torch.cuda.set_device(device_id)

    initialize_model()
    model = model_dict["model"].to(device)
    model.eval()
    preprocess = model_dict["preprocess_val"]

    # Load checkpoint weights once
    ckpt_path  = hf_hub_download("xswu/HPSv2", hps_version_map[self.hps_version])
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    tokenizer  = get_tokenizer("ViT-H-14")

    total = 0.0
    for i in tqdm.tqdm(range(0, len(imgs_gen), batch_size), disable=not verbose):
      images = torch.stack([preprocess(to_pil_image(imgs_gen[j])) for j in range(i,min(len(imgs_gen), i+batch_size))]).to(device)
      # images = self._hps_preprocess(imgs_gen[i:i+batch_size]).to(device)
      text  = tokenizer(prompts[i:i+batch_size]).to(device)
      feats = model(images, text)
      score = (feats["image_features"] @ feats["text_features"].T).diagonal().sum().item()
      total += score

    return total / len(imgs_gen)
  
  def _hps_preprocess(self, images):
    images = resize(images, size=224, antialias=True)
    images = center_crop(images, output_size=224)
    images = normalize(images, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    return images
      
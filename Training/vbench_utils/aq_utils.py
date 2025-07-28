import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import torch.nn.functional as F

try:
  from torchvision.transforms import InterpolationMode
  BICUBIC = InterpolationMode.BICUBIC
  BILINEAR = InterpolationMode.BILINEAR
except ImportError:
  from PIL import Image
  BICUBIC = Image.BICUBIC
  BILINEAR = Image.BILINEAR


def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        # x is now in [-1,1]; map to [0,1]
        transforms.Lambda(lambda x: (x + 1.0) / 2.0),
        # then normalize exactly as CLIP expects
        Normalize(
          mean=(0.48145466, 0.4578275, 0.40821073),
          std =(0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def aq_loss(aesthetic_model, clip_model, video_list, device=None, batch_size=32):
  device = video_list[0].device if device is None else device
  aesthetic_model.eval()
  clip_model.eval()

  video_results = []
  for video in video_list:
    images = video.permute(0, 3, 1, 2).contiguous()
    image_transform = clip_transform(224)

    aesthetic_losses_list = []
    for i in range(0, len(images), batch_size):
      image_batch = images[i:i + batch_size]
      image_batch = image_transform(image_batch)
      image_batch = image_batch.to(device)

      with torch.no_grad():
        image_feats = clip_model.encode_image(image_batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aesthetic_losses = aesthetic_model(image_feats).squeeze(dim=-1)

      aesthetic_losses_list.append(aesthetic_losses)

    aesthetic_losses = torch.cat(aesthetic_losses_list, dim=0) / 10
    video_results.append(1 - aesthetic_losses.mean())

  return torch.tensor(video_results)




def aq_l1_loss(clip_model, video_student_list, video_teacher_list, device=None, batch_size=32):
  device = video_student_list[0].device if device is None else device
  clip_model.eval()

  video_results = []
  for ivid in range(len(video_student_list)):

    video_student = video_student_list[ivid]
    video_teacher = video_teacher_list[ivid]

    images_student = video_student.permute(0, 3, 1, 2).contiguous()
    images_teacher = video_teacher.permute(0, 3, 1, 2).contiguous()

    image_transform = clip_transform(224)

    aq_l1_list = []
    for i in range(0, len(images_student), batch_size):
      image_batch_student = images_student[i:i + batch_size]
      image_batch_teacher = images_teacher[i:i + batch_size]

      image_batch_student = image_transform(image_batch_student).to(device)
      image_batch_teacher = image_transform(image_batch_teacher).to(device)

      with torch.no_grad():
        image_feats_student = clip_model.encode_image(image_batch_student).to(torch.float32)
        image_feats_teacher = clip_model.encode_image(image_batch_teacher).to(torch.float32)
        aq_l1_losses = F.l1_loss(image_feats_student, image_feats_teacher, reduction='none')
        aq_l1_losses = aq_l1_losses.mean(dim=list(range(1,len(aq_l1_losses.shape)))) # [batch_size]

      aq_l1_list.append(aq_l1_losses)

    aq_l1_losses = torch.cat(aq_l1_list, dim=0)
    video_results.append(aq_l1_losses.mean())

  return torch.tensor(video_results)
import torch
from torchvision import transforms
import torch.nn.functional as F

def iq_transform(images, preprocess_mode='shorter'):
    """
    Resize and normalize images, handling both [0,255] uint8 and [0,1] float inputs.
    Args:
        images (torch.Tensor): Input tensor of shape (B, C, H, W).
        preprocess_mode (str): One of 'shorter', 'shorter_centercrop', 'longer', or 'None'.
    Returns:
        torch.Tensor: Resized and normalized image tensor in [0,1].
    """
    # -- Resize based on mode --
    if preprocess_mode.startswith('shorter'):
      _, _, h, w = images.size()
      if min(h, w) > 512:
        scale = 512.0 / min(h, w)
        images = transforms.Resize(size=(int(h * scale), int(w * scale)), antialias=False)(images)
        if preprocess_mode == 'shorter_centercrop':
          images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
      _, _, h, w = images.size()
      if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        images = transforms.Resize(size=(int(h * scale), int(w * scale)), antialias=False)(images)

    elif preprocess_mode == 'None': pass
    else:
      raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")

    # -- Normalize to [0,1] --
    images = images.float()/2.0 + 0.5
    return images

def iq_loss(model, video_list, device=None, **kwargs):
  device = video_list[0].device if device is None else device

  if 'imaging_quality_preprocessing_mode' not in kwargs: preprocess_mode = 'longer'
  else: preprocess_mode = kwargs['imaging_quality_preprocessing_mode']

  video_results = []
  for video in video_list:
    # we need just to make uint8 tensor (T, C, H, W). now it is  # t h w c
    images = video.permute(0, 3, 1, 2).contiguous()
    images = iq_transform(images, preprocess_mode)
    acc_loss_video = 0.
    for i in range(len(images)):
      frame = images[i].unsqueeze(0).to(device)
      loss = model(frame).squeeze()
      acc_loss_video += 1 - loss/100
    video_results.append(acc_loss_video/len(images))
  return torch.tensor(video_results)



def get_features_iq(iq_model, x):
  features = {}
  def grab_before_head(module, inputs):
    features['feat'] = inputs[0]

  hook_handle = iq_model.head.register_forward_pre_hook(grab_before_head)

  logits = iq_model(x)
  teacher_feats = features['feat']   # shape [B, 384]

  hook_handle.remove()
  return teacher_feats


def iq_l1_loss(model, video_student_list, video_teacher_list, device=None, **kwargs):
  device = video_student_list[0].device if device is None else device

  if 'imaging_quality_preprocessing_mode' not in kwargs: preprocess_mode = 'longer'
  else: preprocess_mode = kwargs['imaging_quality_preprocessing_mode']

  video_results = []
  for ivid in range(len(video_student_list)):

    video_student = video_student_list[ivid]
    video_teacher = video_teacher_list[ivid]

    images_student = video_student.permute(0, 3, 1, 2).contiguous()
    images_teacher = video_teacher.permute(0, 3, 1, 2).contiguous()

    images_student = iq_transform(images_student, preprocess_mode)
    images_teacher = iq_transform(images_teacher, preprocess_mode)

    iq_l1_loss = 0.
    for i in range(len(images_student)):
      frame_student = images_student[i].unsqueeze(0).to(device)
      frame_teacher = images_teacher[i].unsqueeze(0).to(device)

      features_student = get_features_iq(model, frame_student)
      features_teacher = get_features_iq(model, frame_teacher)

      loss = F.l1_loss(features_student, features_teacher)
      iq_l1_loss += loss

    video_results.append(iq_l1_loss/len(images_student))
  return torch.tensor(video_results)
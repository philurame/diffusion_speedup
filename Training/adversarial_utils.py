import torch
import torch.nn.functional as F
from unet_D import Discriminator

# G
class LADD:
  def __init__(self, config, device):

    # set discriminator
    model_path_name = config['model_path_name']
    self.disc = Discriminator(model_path_name).to(device)
    self.device = device

    # set noise schedule
    beta_schedule = config['beta_schedule']
    beta_start = config['beta_start']
    beta_end = config['beta_end']
    num_train_timesteps = config['num_train_timesteps']
    if beta_schedule == "linear":
      self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
      self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    if config.get("rescale_betas_zero_snr", False):
      self.alphas_cumprod[-1] = 2**(-24)

  def add_noise(self,original_samples,noise,timesteps):
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
      sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

  def G_loss(self, student_latents_out, teacher_latents_out, prompt_embeds, recon_loss_type, **kwargs):
    self.disc.eval()

    bsz = student_latents_out.shape[0]
    timesteps_D = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    # timesteps_D = torch.sigmoid(torch.distributions.Normal(loc=1, scale=1).sample([bsz])).to(self.device)*999

    noised_student = self.add_noise(student_latents_out, torch.randn_like(student_latents_out), timesteps_D)

    # adv loss
    pred_fake = self.disc(noised_student, timesteps_D, encoder_hidden_states=prompt_embeds)
    adv_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))

    #recon loss (either LPIPS/L1/latent_L1 or None)
    if recon_loss_type == 'LATENT-L1':
      recon_loss = F.smooth_l1_loss(student_latents_out, teacher_latents_out)
    elif recon_loss_type == 'L1':
      student_imgs = kwargs['student_imgs']
      teacher_imgs = kwargs['teacher_imgs']
      recon_loss = F.smooth_l1_loss(student_imgs, teacher_imgs)
    else:
      raise NotImplementedError

    #total loss (sum over provided batch)
    return adv_loss, recon_loss
  
  def D_loss(self, student_latents_out, teacher_latents_out, prompt_embeds):
    self.disc.train()
    
    bsz = student_latents_out.shape[0]

    timesteps_D_fake = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    timesteps_D_real = torch.randint(low=0, high=701, size=(bsz,), device=self.device)

    noised_student = self.add_noise(student_latents_out, torch.randn_like(student_latents_out), timesteps_D_fake)
    noised_teacher = self.add_noise(teacher_latents_out, torch.randn_like(teacher_latents_out), timesteps_D_real)

    pred_fake = self.disc(noised_student, timesteps_D_fake, encoder_hidden_states=prompt_embeds)
    pred_true = self.disc(noised_teacher, timesteps_D_real, encoder_hidden_states=prompt_embeds)
    
    #calculate losses for fake and real data
    loss_gen  = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    loss_real = F.binary_cross_entropy_with_logits(pred_true, torch.ones_like(pred_true))

    return loss_gen, loss_real
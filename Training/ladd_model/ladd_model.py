import torch
import torch.nn.functional as F
from unet_D import Discriminator


# G
class LADD:
  def __init__(self, config, device, freeze):

    # set discriminator
    model_path_name = config['model_path_name']
    self.disc = Discriminator(model_path_name, freeze=freeze).to(device)
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
    teacher_latents_out = teacher_latents_out.detach()

    bsz = student_latents_out.shape[0]
    timesteps_D_fake = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    timesteps_D_real = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    # timesteps_D_fake = (torch.sigmoid(torch.distributions.Normal(1.,1.).sample([bsz])) * 999).to(self.device).long()
    # timesteps_D_real = (torch.sigmoid(torch.distributions.Normal(1.,1.).sample([bsz])) * 999).to(self.device).long()

    noised_student = self.add_noise(student_latents_out, torch.randn_like(student_latents_out), timesteps_D_fake)
    noised_teacher = self.add_noise(teacher_latents_out, torch.randn_like(teacher_latents_out), timesteps_D_real)

    fake_logits = self.disc(noised_student, timesteps_D_fake, encoder_hidden_states=prompt_embeds)
    real_logits = self.disc(noised_teacher, timesteps_D_real, encoder_hidden_states=prompt_embeds)
    
    adv_loss = torch.nn.functional.softplus(-(fake_logits-real_logits))

    recon_loss = F.smooth_l1_loss(student_latents_out, teacher_latents_out, reduction='none')
    recon_loss = recon_loss.mean(dim=list(range(1,len(recon_loss.shape))))

    return adv_loss, recon_loss
  
  def D_loss(self, student_latents_out, teacher_latents_out, prompt_embeds, gamma=0.2, is_train=True):
    self.disc.train()

    is_train = False
    if is_train:
      student_latents_out = student_latents_out.detach().requires_grad_(True)
      teacher_latents_out = teacher_latents_out.detach().requires_grad_(True)
    
    bsz = student_latents_out.shape[0]

    timesteps_D_fake = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    timesteps_D_real = torch.randint(low=0, high=701, size=(bsz,), device=self.device)
    # timesteps_D_fake = (torch.sigmoid(torch.distributions.Normal(1.,1.).sample([bsz])) * 999).to(self.device).long()
    # timesteps_D_real = (torch.sigmoid(torch.distributions.Normal(1.,1.).sample([bsz])) * 999).to(self.device).long()

    noised_student = self.add_noise(student_latents_out, torch.randn_like(student_latents_out), timesteps_D_fake)
    noised_teacher = self.add_noise(teacher_latents_out, torch.randn_like(teacher_latents_out), timesteps_D_real)

    fake_logits = self.disc(noised_student, timesteps_D_fake, encoder_hidden_states=prompt_embeds)
    real_logits = self.disc(noised_teacher, timesteps_D_real, encoder_hidden_states=prompt_embeds)

    if is_train:
      R1Penalty = self.ZeroCenteredGradientPenalty(noised_teacher, real_logits)
      R2Penalty = self.ZeroCenteredGradientPenalty(noised_student, fake_logits)
    else:
      R1Penalty = torch.zeros(noised_student.shape, device=self.device)
      R2Penalty = torch.zeros(noised_student.shape, device=self.device)
    
    logits_diff = real_logits - fake_logits

    adv_loss = torch.nn.functional.softplus(-logits_diff)
    penalty  = (gamma / 2) * (R1Penalty + R2Penalty)

    
    tol = 1e-5
    signs = torch.where(
        logits_diff.abs() < tol,          # condition
        torch.zeros_like(logits_diff),    # if true: neutral
        logits_diff.sign()                # else: usual ±1
    )
    sign_accuracy = signs.mean()/2+1/2

    discriminator_loss = adv_loss + penalty
    return discriminator_loss, adv_loss, penalty, sign_accuracy

  def ZeroCenteredGradientPenalty(self, Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=False, retain_graph=True)
    return Gradient.square().sum([1, 2, 3])
  
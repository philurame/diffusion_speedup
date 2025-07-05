import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from lib.generate_decode import seed_everything
from training_cachers.loss_cachers import PatchedLPIPS

def init_logits(args):
    if args.init_logits == 'ones':
        logits = torch.ones(               
            args.student_nfe, dtype=torch.float32, 
            device=args.device, 
            requires_grad=True
        ) 
        
    if  args.init_logits == 'randn':
        logits = torch.randn(               
            args.student_nfe, dtype=torch.float32, 
            device=args.device, 
            requires_grad=True
        )
        
    if args.init_logits == 'zeros':
        logits = torch.zeros(               
            args.student_nfe, dtype=torch.float32, 
            device=args.device, 
            requires_grad=True
        )
        
    if args.init_logits == 'deepcache-3':
        logits = torch.ones(               
            args.student_nfe, dtype=torch.float32, 
            device=args.device
        )
        # пересчитываемые логиты должны быть поменьше
        logits[2::3] *= 0.9
        logits.requires_grad_(True)
        
    return logits

def sample_exp(logits, inference=False):
    """
    Сэмплирует значения по Exp-Min Trick.
    
    Для каждой i-ой компоненты получаем случайную величину G_i = g_i + l_i,
    где l_i - это i-й логит; g_i=log(-log(U_i)), U_i ~ Uniform(0, 1) - шум из стандартного распределения Гумбеля
    """
    if inference:
        return torch.exp(logits)

    uniform_noise = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    exp_noise = torch.log(-torch.log(uniform_noise))
    
    sampled_values = torch.exp(logits + exp_noise)
    sampled_values.requires_grad_(True)
    
    return sampled_values

def top_k_log_prob(logits, perturbed_logits, k):
    """
    Вычисляет log-prob для первых k элементов упорядоченной последовательности логитов.

    log P(t) = log P(t_1, ..., t_k) для засэмплированной последовательности t.
    
    P(t) = ∏_{i=1..k} [ exp(θ_{t_i}) / (Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(θ_j)) ]
    
    log P(t) = Σ_{i=1..k} [ exp(θ_{t_i}) - log(Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(θ_j)) ], где θ - это наши `logits`.

    Args:
        logits - исходный вектор логитов
        perturbed_logits - зашумленные логиты из sample_gumbel_max
        k - количество логитов, для которых считаем log-prob

    Returns:
        tuple(
            Индексы первых k элементов в отсортированной последовательности,
            log P(t_1, ..., t_k)
        )
    """
    num_total_steps = logits.shape[0] # (N, )
    
    # Получаем упорядоченную последовательность t = (t_1, ..., t_k), что эквивалетно взятию argmin k раз.
    indices = torch.argsort(perturbed_logits)
    top_k_indices = indices[:k] # (k, )
    
    # Создаем бинарную маску, где bin_mask[i, j] = 1, если argmins[i] = j
    all_indices = torch.arange(num_total_steps, device=logits.device)
    bin_mask = (top_k_indices[:, None] == all_indices[None, :]) # (k, N)
    
    # Вычисляем числитель (выбираем нужный логит для каждого шага i)
    log_numerator = - (logits[None, :] * bin_mask).sum(dim=-1) # (k,)

    # Вычисляем знаменатель
    # Создаем маску, которая на шаге i будет выключать уже выбранные элементы t_1, ..., t_{i-1}.
    exclusion_mask = torch.cumsum(bin_mask, dim=0)[:-1, :] # (k-1, N)
    # Добавляем строчку нулей, которая говорить, что на шаге 1 мы ничего не исключаем
    exclusion_mask = torch.cat([torch.zeros(1, num_total_steps, device=logits.device), exclusion_mask], dim=0) # (k, N)    
    # Применяем маску: добавляем -inf к уже выбранным логитам, чтобы они не участвовали в logsumexp.
    exclusion_mask[exclusion_mask > 0] = float('inf')
    masked_logits = -logits[None, :] - exclusion_mask
    log_denominator = torch.logsumexp(masked_logits, dim=-1) # (k,) 
    
    # Собираем все вместе
    log_prob = (log_numerator - log_denominator).sum() 

    # Мы передаем в cacher шаги, которые пропускаем
    return indices[k:] + 1, log_prob


def log_train(logits, loss, logprobs, before_baseline, grad_mean, scheduler, step, args):
    for i in range(len(logits)):
        wandb.log({
            f'logit #{i+1}': logits[i].item()
        }, step=step)
        
    grad_norm = logits.grad.data.norm(2).item()
    grad_mean.append(grad_norm)
    
    wandb.log({
        'train loss': loss.item(),
        f'train {args.metric} before baseline': before_baseline.item(),
        'log prob': logprobs.mean().item(),
        'grad norm': grad_norm,
        'grad moving average': np.nanmean(grad_mean),
        'lr': scheduler.get_last_lr()[0]
    }, step=step)


def log_validation(pipe, helper, logits, noise, val_dataloader, baseline_imgs, metric, step, args, display_k=4):
    
    def concat_images(orig_imgs, gen_imgs, baseline_imgs):
        col_orig     = torch.cat(orig_imgs.unbind(0), dim=1)
        col_gen      = torch.cat(gen_imgs.unbind(0), dim=1)
        col_baseline = torch.cat(baseline_imgs.unbind(0), dim=1)
        grid = torch.cat([col_orig, col_gen, col_baseline], dim=2)
        img = grid.mul(0.5).add(0.5).clamp(0, 1)
        return img.permute(1, 2, 0).cpu().numpy()
    
    mode_logits = sample_exp(logits, inference=True)
    ts = torch.argsort(mode_logits)[args.num_steps:] + 1
    
    metric_value = 0.
    count = 0
    
    with torch.no_grad():
        for anns in tqdm(val_dataloader, 'Validation', leave=False):
            with helper.default():
                original = pipe(
                    anns, 
                    num_inference_steps=args.teacher_nfe, 
                    guidance_scale=args.gs,
                    latents=noise,
                    output_type='pt'
                )
                
            with helper.inference(timesteps=ts):
                generated = pipe(
                    anns, 
                    num_inference_steps=args.student_nfe, 
                    guidance_scale=args.gs,
                    latents=noise,
                    output_type='pt'
                )
                
            metric_value += metric.calculate(original, generated).mean().item()
            count += 1
    
    wandb.log({
        
    }, step=step)


    val_images = []
    img1 = concat_images(
        original[:display_k],
        generated[:display_k],
        baseline_imgs[:display_k]
    )
    val_images.append(
        wandb.Image(
            img1,
            caption=f"original vs generated vs {args.baseline_name.lower()}, part 1"
        )
    )
    img2 = concat_images(
        original[-display_k:],
        generated[-display_k:],
        baseline_imgs[-display_k:]
    )
    val_images.append(
        wandb.Image(
            img2,
            caption=f"original vs generated vs {args.baseline_name.lower()}, part 2"
        )
    )
    
    fig, ax = plt.subplots()
    steps = [(t not in ts) for t in range(args.student_nfe + 1)]
    ax.plot([*range(args.student_nfe + 1)], steps)
    ax.set_xticks([*range(args.student_nfe + 1)])
    wandb.log({
        f'val {args.metric}': metric_value / count,
        "val images": val_images,
        "timesteps plot": wandb.Image(fig),
    }, step=step)
    plt.close(fig)

def reinforce_training_loop(
    pipe,
    baseline_pipe,
    train_dataloader,
    val_dataloader,
    metric_name,
    args
):
    
    seed_everything()
    
    helper = pipe.cacher
    if metric_name.lower() == "patched-lpips":
        metric = PatchedLPIPS(device=pipe.device)
        
    latent_size = 1024 // pipe.vae_scale_factor
    val_noise = torch.randn(
        (args.batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
    if args.same_train_noise:
        train_noise = torch.randn(
            (args.batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
    
    num_steps = args.num_steps - 1
    student_nfe = args.student_nfe - 1
    
    logits = init_logits(args) 
    grad_mean = deque([np.nan] * 8, maxlen=8) 

    optim = torch.optim.Adam([logits], args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
    
    print(f"The beginning of the baseline generating {args.baseline_name}")
    last_val_anns = list(val_dataloader)[-1]
    baseline_imgs = baseline_pipe(
        last_val_anns, 
        num_inference_steps=args.teacher_nfe, 
        guidance_scale=args.gs,
        latents=val_noise,
        output_type='pt'
    )
    del baseline_pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"The end of the baseline generating {args.baseline_name}")
    
    for epoch in tqdm(range(args.epochs), 'Epochs'):

        log_validation(
            pipe, helper, logits, 
            val_noise, val_dataloader, baseline_imgs,
            metric, epoch * len(train_dataloader), args
        )
        
        for step, anns in tqdm(enumerate(train_dataloader), 'Train loader', leave=False):
            step = epoch * len(train_dataloader) + step
            
            if not args.same_train_noise:
                train_noise = torch.randn(
                    (args.batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
                
                
            optim.zero_grad()
            with helper.default():
                original = pipe(
                    anns, num_inference_steps=args.teacher_nfe, guidance_scale=args.gs,
                    latents=train_noise,
                    output_type='pt'
                )
                
            metrics = torch.empty((args.num_samples, args.batch_size), device=args.device) # [num_samples, batch_size]
            logprobs = torch.zeros(args.num_samples, device=args.device)
            
            for i in tqdm(range(args.num_samples), 'Reinforce steps', leave=False):
                perturbed_logits = sample_exp(logits, inference=False)
                
                ts, log_prob = top_k_log_prob(logits, perturbed_logits, num_steps)

                with helper.inference(timesteps=ts):
                    generated = pipe(
                        anns, num_inference_steps=student_nfe, guidance_scale=args.gs,
                        latents=train_noise,
                        output_type='pt'
                    )
                
                logprobs[i] = log_prob
                metrics[i, :] = metric.calculate(original, generated).detach()

            
            metrics_mean = metrics.mean(dim=0)
            metrics_corrected = (metrics - metrics_mean[None, :]).detach()
            
            loss = (metrics_corrected * logprobs[:, None]).mean() * (args.num_samples) / (args.num_samples - 1)
            

            loss.backward()
            optim.step()
            
            log_train(
                logits, loss, logprobs, metrics_mean.mean(), 
                grad_mean, scheduler, step, args
            )
        
        scheduler.step()
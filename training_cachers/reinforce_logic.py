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


def concat_images(images: list[torch.Tensor]) -> wandb.Image:
    max_h = max(img.shape[1] for img in images)
    padded_images = []
    for img in images:
        pad_height = max_h - img.shape[1]
        if pad_height > 0:
            pad_tensor = torch.full((3, pad_height, img.shape[2]), -1.0, 
                                   device=img.device, dtype=img.dtype)
            img = torch.cat([img, pad_tensor], dim=1)
        padded_images.append(img)
    concat_tensor = torch.cat(padded_images, dim=2)
    return concat_tensor.mul(0.5).add(0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def sample_exp(logits, inference=True, eps=torch.tensor([0.])):
    '''
    x = log(1-u) * 1 / (-λ), где u = uniformRand(0, 1)
    x = - log(u) / λ

    Tensor 'logits' contains log-means of the exponential distributions
    Parameters of the exponentials can be represented as lambda = exp(-logit)
    
    exp(logit) - log(uniform) 
    uniform \in (0, 1) -> log \in (-inf, 0) -> -log \in (0, +inf)
    чем меньше логит, тем меньше exp, тем больше влияние шуме
    и тем меньше на инференсе значение -> этот индекс будет вначале
    
    то есть нам на самом деле надо искать последние 9 индексов, 
    чтобы у них был логит побольше, чтобы меньше влияние шума, чтобы на инференсе всё было ок
    '''
    if inference:
        return torch.exp(logits)
    
    random = torch.bernoulli(eps)
    
    if not random:
        sampled = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
        sampled = logits + torch.log(-torch.log(sampled))
    else:
        sampled = logits[torch.randperm(logits.shape[0])]
    
    exp = torch.exp(sampled)        
    exp.requires_grad_(True)
    return exp


def top_k_log_prob(logits, exp, k):
    # exp.shape == logits.shape: (length,), 
    # length - размер ренжда параметров
    length = exp.shape[0]
    possible_values = torch.arange(length, device=exp.device)
    
    values = torch.argsort(exp)[:k]

    # матрица, где True стоит в строке k и столбце l, если argmins[k] == l
    bin_values = values[:, None] == possible_values[None, :]        # (k, length)
    
    mask = torch.zeros(k, length, device=logits.device)
    mask[1:, :] = torch.cumsum(bin_values, dim=0)[:-1, :]
    mask[mask > 0] = float('inf')

    # просто матрица логитов
    logits_min = (logits[None, :] * bin_values).sum(dim=-1)          # (k,)
    masked_logits = logits[None, :] + mask                           # (k, length)

    log_prob = -logits_min - torch.logsumexp(-masked_logits, dim=-1)
    log_prob = log_prob.sum()
    
    return torch.argsort(exp)[k:] + 1, log_prob
    # в итоге возвращаю не values, а другие таймстепы, 
    # потому что хочу простить задачу и выбирать 9 из 25, 
    # но подавать cacher'у надо timesteps to cache
    
    
def log_validation(pipe, helper, logits, noise, val_dataloader, metric, step, args):
    sampled = sample_exp(logits, inference=True)
    ts = torch.argsort(sampled)[args.k:] + 1    
    
    metric_value = 0
    counter = 0
    
    with torch.no_grad():
        for anns in tqdm(val_dataloader, 'Validation', leave=False):
            with helper.default():
                original = pipe(
                    anns, num_inference_steps=args.teacher_nfe, guidance_scale=args.gs,
                    latents=noise,
                    output_type='pt'
                )
            # original = metric.preprocess(original).to(args.device)
                
            with helper.inference(timesteps=ts):
                generated = pipe(
                    anns, num_inference_steps=args.student_nfe, guidance_scale=args.gs,
                    latents=noise,
                    output_type='pt'
                )
                
            metric_value += metric.calculate(original, generated).item()
            counter += 1
    
    wandb.log({
        f'val {args.metric}': metric_value / counter
    }, step=step)
    
    
    images = []
    for i, (orig, gen) in enumerate(zip(original, generated)):
        img = wandb.Image(
            concat_images([orig, gen]),
            caption=f"image #{i}"
        )
        images.append(img)
    wandb.log({"Images: orig VS gen": images}, step=step)
        
    
    fig, ax = plt.subplots()
    steps = [(t not in ts) for t in range(args.student_nfe + 1)]
    ax.plot([*range(args.student_nfe + 1)], steps)
    ax.set_xticks([*range(args.student_nfe + 1)])
    wandb.log({
        "timesteps plot": wandb.Image(fig),
    }, step=step)
    plt.close(fig)
    
    
def log_train(logits, losses, log_probs, before_baseline, after_baseline, grad_mean, scheduler, step, args):
    for i in range(args.student_nfe):
        wandb.log({
            f'logit #{i+1}': logits[i].item()
        }, step=step)
        
    grad_norm = logits.grad.data.norm(2).item()
    grad_mean.append(grad_norm)
    
    wandb.log({
        'train loss': losses.item(),
        f'train {args.metric} before baseline': before_baseline.item(),
        f'train {args.metric}': after_baseline.item(),
        'log prob': log_probs.mean().item(),
        'grad norm': grad_norm,
        'grad moving average': np.nanmean(grad_mean),
        'lr': scheduler.get_last_lr()[0]
    }, step=step)
    

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
        # все логиты большие, 
        # те, которые пересчитываем -- поменьше
        logits = torch.ones(               
            args.student_nfe, dtype=torch.float32, 
            device=args.device
        ) 
        logits[2::3] *= 0.9
        logits.requires_grad_(True)
        
    return logits

    
def reinforce_training_loop(pipe, train_dataloader, val_dataloader, metric_name, args):
    
    if metric_name.lower() == "patched-lpips":
        metric = PatchedLPIPS(device=pipe.device)
    
    seed_everything()
    ss = 1024 // pipe.vae_scale_factor
    
    helper = pipe.cacher
    
    val_noise = torch.randn((args.batch_size, 4, ss, ss), dtype=pipe.dtype, device=args.device)
    if args.same_train_noise:
        train_noise = torch.randn((args.batch_size, 4, ss, ss), dtype=pipe.dtype, device=args.device)
    
    args.k -= 1                         # top k (сколько мы хотим кешировать)
    args.student_nfe -= 1               # в целом из какого рэнджа берем значения
    
    logits = init_logits(args)          # log-means
    
    grad_mean = deque([np.nan] * 8, maxlen=8) 
    
    optim = torch.optim.Adam([logits], args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
    
    for epoch in tqdm(range(args.epochs), 'Epochs'):
        log_validation(pipe, helper, logits, val_noise, val_dataloader, metric, epoch * len(train_dataloader), args)
        
        for step, anns in tqdm(enumerate(train_dataloader), 'Train loader', leave=False):
            step = epoch * len(train_dataloader) + step
            batch_size = len(anns)
            
            if not args.same_train_noise:
                train_noise = torch.randn((args.batch_size, 4, ss, ss), dtype=pipe.dtype, device=args.device)
            
            optim.zero_grad()
            with helper.default():
                original = pipe(
                    anns, num_inference_steps=args.teacher_nfe, guidance_scale=args.gs,
                    latents=train_noise,
                    output_type='pt'
                )
            # original = metric.preprocess(original).to(args.device)# TODO: fix with patched lpips

            metrics = torch.empty((batch_size, args.m), device=args.device)
            log_probs = []
            
            for i in tqdm(range(args.m), 'Reinforce steps', leave=False):
                sampled = sample_exp(logits, inference=False)
                # короче мне тут надо переписать на leave-one-out samples size сразу
                # и можно сразу скор посчитать матрично, а не итерироваться...................
                # а потом уже итерироваться, чтобы лосс посчитать
                ts, log_prob = top_k_log_prob(logits, sampled, args.k)
                log_probs.append(log_prob)
                
                with helper.inference(timesteps=ts):
                    generated = pipe(
                        anns, num_inference_steps=args.student_nfe, guidance_scale=args.gs,
                        latents=train_noise,
                        output_type='pt'
                    )
                    
                # losses[:, i] = metric.calculate(original_patches, generated)  # хотим его минимизировать
                metrics[:, i] = metric.calculate(original, generated).detach()
            
            before_baseline = metrics.detach().clone().mean()    
            
            metrics = metrics - metrics.mean(dim=-1)[:, None]
            metrics /= (args.m - 1)
            
            after_baseline = metrics.detach().clone().mean()    

            log_probs = torch.stack(log_probs).to(args.device)
            loss = (metrics * log_probs).mean()
            
            # losses.backward(logits)
            loss.backward()
            log_train(
                logits, loss, log_probs,
                before_baseline, after_baseline, 
                grad_mean, scheduler, step, args
            )
            
            optim.step()
            
            if args.logits_clipping is not None:
                with torch.no_grad():
                    logits.clamp_(-args.logits_clipping, args.logits_clipping)
            
        scheduler.step()
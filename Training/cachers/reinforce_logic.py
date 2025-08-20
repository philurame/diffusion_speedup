import torch
import numpy as np
# import wandb
from neptune.types import File
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import gc
import sys
import os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from Training.models import seed_everything
from Training.cachers.loss_cachers import PatchedLPIPS

def init_logits(args):
    
    temp_student_nfe = args.student_nfe - 1 # we never cache first step
    if args.init_logits == 'ones':
        logits = torch.ones(               
            temp_student_nfe, dtype=torch.float32, 
            device=args.device, requires_grad=True
        ) 
    elif  args.init_logits == 'randn':
        logits = torch.randn(               
            temp_student_nfe, dtype=torch.float32, 
            device=args.device, requires_grad=True
        )
    elif args.init_logits == 'zeros':
        logits = torch.zeros(               
            temp_student_nfe, dtype=torch.float32, 
            device=args.device, requires_grad=True
        ) 
    elif args.init_logits == 'deepcache-3':
        logits = torch.ones(               
            temp_student_nfe, dtype=torch.float32, 
            device=args.device
        )
        logits[2::3] *= 0.9 # пересчитываемые логиты должны быть поменьше
        logits.requires_grad_(True)
    else:
        raise ValueError(f"Unknown init_logits type: {args.init_logits}")
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


def log_train(logits, loss, logprobs, before_baseline, grad_mean, scheduler, step, args, run):
    # for i in range(len(logits)):
    #     wandb.log({
    #         f'logit #{i+1}': logits[i].item()
    #     }, step=step)
    for i in range(len(logits)):
        # run[f'logits/logit_{i+1}'].log(logits[i].item(), step=step)
        run[f'logits/logit_{i+1}'].append(logits[i].item(), step=step)
        
    grad_norm = logits.grad.data.norm(2).item()
    grad_mean.append(grad_norm)
    
    # wandb.log({
    #     'train loss': loss.item(),
    #     f'train {args.metric} before baseline': before_baseline.item(),
    #     'log prob': logprobs.mean().item(),
    #     'grad norm': grad_norm,
    #     'grad moving average': np.nanmean(grad_mean),
    #     'lr': scheduler.get_last_lr()[0]
    # }, step=step)
    # run["train/loss"].append(float(loss.item()), step=step)
    # run[f"train/{args.metric}_before_baseline"].append(float( before_baseline.item()), step=step)
    # run["train/log_prob_mean"].append(float(logprobs.mean().item()), step=step)
    # run["train/grad_norm"].append(float(grad_norm), step=step)
    # run["train/grad_moving_average"].append(float(np.nanmean(grad_mean)), step=step)
    # run["train/lr"].append(float(scheduler.get_last_lr()[0]), step=step)

    run["train"].append({
        'train loss': loss.item(),
        f'train {args.metric} before baseline': before_baseline.item(),
        'log prob': logprobs.mean().item(),
        'grad norm': grad_norm,
        'grad moving average': np.nanmean(grad_mean),
        'lr': scheduler.get_last_lr()[0]
    }, step=step)


def log_validation(pipe, helper, logits, val_noise, val_dataloader, teacher_val_images, baseline_imgs, metric, step, args, run, display_k=4):
    
    def concat_images(orig_imgs, gen_imgs, baseline_imgs):
        col_orig     = torch.cat(orig_imgs.unbind(0), dim=1).cpu()
        col_gen      = torch.cat(gen_imgs.unbind(0), dim=1).cpu()
        col_baseline = torch.cat(baseline_imgs.unbind(0), dim=1).cpu()
        grid = torch.cat([col_orig, col_gen, col_baseline], dim=2)
        img = grid.mul(0.5).add(0.5).clamp(0, 1)
        return img.permute(1, 2, 0).cpu().numpy()
    
    mode_logits = sample_exp(logits, inference=True)
    ts = torch.argsort(mode_logits)[args.num_steps:] + 1
    
    metric_value = 0.
    count = 0
    
    with torch.no_grad():
        for batch_idx, anns in tqdm(enumerate(val_dataloader), 'Validation', leave=False):
            batch_start = batch_idx * args.val_batch_size
            batch_end = batch_start + args.val_batch_size
            original = teacher_val_images[batch_start:batch_end].to(pipe.device)

            with helper.inference(timesteps=ts):
                generated = pipe(
                    anns, 
                    num_inference_steps=args.student_nfe, 
                    guidance_scale=args.gs,
                    latents=val_noise[:len(anns)],
                    output_type='pt'
                )
            metric_value += metric.calculate(original, generated).mean().item()
            count += 1

    baseline = baseline_imgs[-len(original):]

    # val_images = []
    img1 = concat_images(
        original[:display_k],
        generated[:display_k],
        baseline[:display_k]
    )
    # val_images.append(
    #     wandb.Image(
    #         img1,
    #         caption=f"original vs generated vs {args.baseline_name.lower()} 1, step {step}"
    #     )
    # )
    img2 = concat_images(
        original[-display_k:],
        generated[-display_k:],
        baseline[-display_k:]
    )
    # val_images.append(
    #     wandb.Image(
    #         img2,
    #         caption=f"original vs generated vs {args.baseline_name.lower()} 2, step {step}"
    #     )
    # )

    img1_to_log = (img1 * 255).round().astype(np.uint8)
    img2_to_log = (img2 * 255).round().astype(np.uint8)

    def resize_image(img_array, target_size=(256, 256)):
        img = Image.fromarray(img_array)
        resized_img = img.resize(target_size, resample=Image.LANCZOS)
        return np.array(resized_img)

    resized_img1 = resize_image(img1_to_log, target_size=(1536, 2048))
    resized_img2 = resize_image(img2_to_log, target_size=(1536, 2048))

    run['val/images/variant_1'].append(
        File.as_image(resized_img1),
        description=f"original vs generated vs {args.baseline_name.lower()} 1, step {step}",
        step=step
    )

    run['val/images/variant_2'].append(
        File.as_image(resized_img2),
        description=f"original vs generated vs {args.baseline_name.lower()} 2, step {step}",
        step=step
    )

    fig, ax = plt.subplots()
    steps = [t not in ts for t in range(args.student_nfe)]
    ax.plot(range(args.student_nfe), steps)
    ones_positions = [i for i, val in enumerate(steps) if val == 1]
    for pos in ones_positions:
        ax.axvline(x=pos, ymax=1, linestyle='--', color='green', alpha=0.7)
        ax.scatter(pos, 1, color='red', s=50, zorder=10)
    ax.set_xticks(range(args.student_nfe))
    ax.set_title(f"Timesteps, step {step}")
    # wandb.log({
    #     f'val {args.metric}': metric_value / count,
    #     "val images": val_images,
    #     "timesteps plot": wandb.Image(fig, caption=f"Timesteps, step {step}"),
    # }, step=step)
    run['val/timesteps_plot'].append(fig, step=step)
    plt.close(fig)

    run[f'val/{args.metric}'].append(float(metric_value / count), step=step)


def generate_data(pipe, baseline_pipe, train_dataloader, val_dataloader, train_noise, val_noise, args):

    teacher_train_data_path = os.path.join(ROOT, "DATA", "cachers", "train_data", f"teacher_train_{args.model_name}_{args.solver}_{args.scheduler}_{args.max_samples}.pt")
    if not os.path.exists(teacher_train_data_path):
        with torch.no_grad():
            teacher_outputs_train = []
            for anns in tqdm(train_dataloader, 'Generating teacher train data', leave=False):
                original_train = pipe(
                    anns, 
                    num_inference_steps=args.teacher_nfe,
                    guidance_scale=args.gs,
                    latents=train_noise[:len(anns)],
                    output_type='pt'
                )
                teacher_outputs_train.append(original_train)
            teacher_outputs_train = torch.cat(teacher_outputs_train, dim=0)        
            torch.save(
                teacher_outputs_train.detach().cpu(), 
                teacher_train_data_path
            )
            print(f"\nTeacher train data saved: {teacher_train_data_path}.\n")
    else:
        print(f"\nFound ready teacher train data: {teacher_train_data_path}.\n")


    teacher_val_data_path = os.path.join(ROOT, "DATA", "cachers", "train_data", f"teacher_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.max_samples}.pt")
    if not os.path.exists(teacher_val_data_path):
        with torch.no_grad():
            teacher_outputs_val = []
            for anns in tqdm(val_dataloader, 'Generating val train data', leave=False):
                original_val = pipe(
                    anns, 
                    num_inference_steps=args.teacher_nfe,
                    guidance_scale=args.gs,
                    latents=val_noise[:len(anns)],
                    output_type='pt'
                )
                teacher_outputs_val.append(original_val)
            teacher_outputs_val = torch.cat(teacher_outputs_val, dim=0)        
            torch.save(
                teacher_outputs_val.detach().cpu(), 
                teacher_val_data_path
            )
            print(f"\nTeacher val data saved: {teacher_val_data_path}.\n")
    else:
        print(f"\nFound ready teacher val data: {teacher_val_data_path}.\n")


    baseline_val_data_path = os.path.join(ROOT, "DATA", "cachers", "train_data", f"{args.baseline_name}_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.max_samples}.pt")
    if not os.path.exists(baseline_val_data_path):
        with torch.no_grad():
            baseline_outputs_val = []

            for anns in tqdm(val_dataloader, 'Generating Baseline Val loader', leave=False):
                original_baseline = baseline_pipe(
                    anns, 
                    num_inference_steps=args.teacher_nfe, 
                    guidance_scale=args.gs,
                    latents=val_noise[:len(anns)],
                    output_type='pt'
                )
                
                baseline_outputs_val.append(original_baseline)

        baseline_outputs_val = torch.cat(baseline_outputs_val, dim=0)        
        torch.save(
            baseline_outputs_val.cpu().detach(), 
            baseline_val_data_path
        )
        print(f"\n{args.baseline_name} val data saved: {baseline_val_data_path}.\n")
    else:
        print(f"\nFound ready {args.baseline_name} val data: {baseline_val_data_path}.\n")

    return teacher_train_data_path, teacher_val_data_path, baseline_val_data_path

def reinforce_training_loop(
    pipe,
    baseline_pipe,
    train_dataloader,
    val_dataloader,
    metric_name,
    args,
    run
):
    
    seed_everything()
    
    helper = pipe.cacher
    latent_size = pipe.unet.config.sample_size
    if metric_name.lower() == "patched-lpips":
        metric = PatchedLPIPS(device=pipe.device)

    val_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"val_noise_{args.val_batch_size}.pt")
    if os.path.exists(val_noise_path):
        val_noise = torch.load(val_noise_path, weights_only=True).to(args.device)
    else:
        val_noise = torch.randn(
            (args.val_batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
        torch.save(
            val_noise, 
            val_noise_path
        )

    if args.same_train_noise:
        train_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"train_noise_{args.train_batch_size}.pt")
        if os.path.exists(train_noise_path):
            train_noise = torch.load(train_noise_path, weights_only=True).to(args.device)
        else:
            train_noise = torch.randn(
                (args.train_batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
            torch.save(
                train_noise, 
                train_noise_path
            )
    
    args.num_steps = args.num_steps - 1
    logits = init_logits(args) 
    grad_mean = deque([np.nan] * 8, maxlen=8) 
    optim = torch.optim.Adam([logits], args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)

    teacher_train_data_path, teacher_val_data_path, baseline_val_data_path = generate_data(
        pipe,
        baseline_pipe,
        train_dataloader,
        val_dataloader,
        train_noise,
        val_noise,
        args
    )
    teacher_train_images = torch.load(teacher_train_data_path, weights_only=True)
    teacher_val_images = torch.load(teacher_val_data_path, weights_only=True)
    baseline_val_images = torch.load(baseline_val_data_path, weights_only=True)

    del baseline_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("\nEverything ready for training!\n")

    
    for epoch in tqdm(range(args.epochs), 'Epochs'):

        if epoch % args.eval_freq ==0:
            log_validation(
                pipe, helper, logits, val_noise, 
                val_dataloader, teacher_val_images, baseline_val_images,
                metric, epoch * len(train_dataloader), args, run
            )
        
        for batch_idx, anns in tqdm(enumerate(train_dataloader), 'Training', leave=False):
            global_step = epoch * len(train_dataloader) + batch_idx
            
            if not args.same_train_noise:
                train_noise = torch.randn(
                    (args.train_batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)
                
                
            optim.zero_grad()
            
            batch_start = batch_idx * args.train_batch_size
            batch_end = batch_start + args.train_batch_size
            original = teacher_train_images[batch_start:batch_end].to(pipe.device)
                
            metrics = torch.empty((args.num_samples, args.train_batch_size), device=args.device) # [num_samples, batch_size]
            logprobs = torch.zeros(args.num_samples, device=args.device)
            
            for i in tqdm(range(args.num_samples), 'Reinforce steps', leave=False):
                perturbed_logits = sample_exp(logits, inference=False)
                
                ts, log_prob = top_k_log_prob(logits, perturbed_logits, args.num_steps)

                with helper.inference(timesteps=ts):
                    generated = pipe(
                        anns, 
                        num_inference_steps=args.student_nfe, 
                        guidance_scale=args.gs,
                        latents=train_noise[:len(anns)],
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
                grad_mean, scheduler, global_step, args, run
            )
        
        scheduler.step()

    log_validation(
        pipe, helper, logits, val_noise, 
        val_dataloader, teacher_val_images, baseline_val_images,
        metric, epoch * len(train_dataloader), args, run
        )
import torch
import numpy as np
import neptune
from neptune.types import File
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import gc
import sys
from PIL import Image
from tqdm import tqdm
from munch import Munch, unmunchify
from collections import deque
from datetime import datetime
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from lib.models.SDXL.sdxl import BaseSDXL
from lib.models.models_utils.caching_timestep_helper import CachingTimestepHelper

from Training.cachers.logit_predictor import load_logit_model, BaseLogitModel
from Training.models import seed_everything
from Training.cachers.loss_cachers import PatchedLPIPS
from Training.cachers.reinforce_logic import sample_exp, top_k_log_prob
from registries import metric_registry

def log_train(
    logit_model: BaseLogitModel, 
    loss: torch.Tensor, 
    logprobs: torch.Tensor, 
    before_baseline_and_reg: torch.Tensor, 
    before_baseline: torch.Tensor, 
    model_grad_mean: deque,
    logits_grad_mean: deque, 
    scheduler: torch.optim.lr_scheduler, 
    step: int, 
    args: Munch, 
    run: neptune.Run 
):
    logits = logit_model.logits
    
    for i in range(len(logits)):
        run[f'logits/logit_{i+1}'].append(logits[i].item(), step=step)
    
    parameters = [
        p for p in logit_model.parameters() 
        if p.grad is not None and p.requires_grad and p is not logits
    ]
    
    model_norm = 0
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        model_norm += param_norm.item() ** 2
    model_norm = model_norm ** 0.5
    
    logits_grad_norm = 0
    if logits.grad is not None:
        logits_grad_norm = logits.grad.data.norm(2).item() ** 2
        logits_grad_norm = logits_grad_norm ** 0.5
    
    model_grad_mean.append(model_norm)
    logits_grad_mean.append(logits_grad_norm)
    
    
    run["train"].append({
        'train loss': loss.item(),
        f'train {args.metric} before baseline and regularization': before_baseline_and_reg.item(),
        f'train {args.metric} before baseline': before_baseline.item(),
        'log prob': logprobs.mean().item(),
        'logits grad norm': logits_grad_norm,
        'logits grad norm moving average': np.nanmean(logits_grad_mean),
        'model grad norm': model_norm,
        'model grad norm moving average': np.nanmean(model_grad_mean),
        'lr': scheduler.get_last_lr()[0]
    }, step=step)


def log_validation(
    pipe: BaseSDXL, 
    helper: CachingTimestepHelper, 
    logit_model: BaseLogitModel, 
    val_noise: torch.Tensor, 
    val_dataloader: DataLoader, 
    teacher_val_images: torch.Tensor, 
    baseline_imgs: torch.Tensor, 
    metric: torch.Tensor, 
    step: int, 
    args: Munch, 
    run: neptune.Run
):
    
    def concat_images(orig_imgs, gen_imgs, baseline_imgs):
        col_orig     = torch.cat(orig_imgs.unbind(0), dim=1)
        col_gen      = torch.cat(gen_imgs.unbind(0), dim=1)
        col_baseline = torch.cat(baseline_imgs.unbind(0), dim=1)
        grid = torch.cat([col_orig, col_gen, col_baseline], dim=2)
        img = grid.mul(0.5).add(0.5).clamp(0, 1)
        return img.permute(1, 2, 0).numpy()
    
    metric_value = 0.
    count = 0
    
    original, generated = [], []
    timesteps = []
    
    with torch.no_grad():
        for batch_idx, anns in enumerate(tqdm(val_dataloader, 'Validation', leave=False)):
            batch_start = batch_idx * args.batch_size
            batch_end = batch_start + args.batch_size
            
            orig = teacher_val_images[batch_start:batch_end].to(pipe.device)
            current_noise = val_noise[batch_start:batch_end]
            
            prompt_embeddings = pipe.encode_prompt(anns, device=args.device)
            logits = logit_model(prompt_embeddings)
            
            mode_logits = sample_exp(logits, inference=True)
            ts_to_skip = torch.argsort(mode_logits)[args.num_steps:] + 1

            with helper.inference(timesteps=ts_to_skip):
                gen = pipe(
                    prompt_embeddings=prompt_embeddings, 
                    num_inference_steps=args.student_nfe, 
                    guidance_scale=args.gs,
                    latents=current_noise,
                    output_type='pt'
                )
            metric_value += metric.calculate(orig, gen).mean().item()
            count += 1
            
            original.append(orig.cpu())
            generated.append(gen.cpu())
            timesteps.append(ts_to_skip)
            
    original = torch.cat(original, dim=0)[-args.logging.display_k:]
    generated = torch.cat(generated, dim=0)[-args.logging.display_k:]
    baseline = baseline_imgs[-args.logging.display_k:]
    
    img = concat_images(original, generated, baseline)

    img = (img * 255).round().astype(np.uint8)

    def resize_image(img_array, target_size=(256, 256)):
        img = Image.fromarray(img_array)
        resized_img = img.resize(target_size, resample=Image.LANCZOS)
        return np.array(resized_img)

    img = resize_image(img, target_size=(512 * 3, args.logging.display_k * 512))

    run['val/images'].append(
        File.as_image(img),
        description=f"original vs generated vs deepcache3, step {step}",
        step=step
    )
    
        
    def draw_plot(ax, timesteps):
        steps = [t not in timesteps for t in range(args.student_nfe)]
        ones_positions = [i for i, val in enumerate(steps) if val == 1]
        
        ax.plot(range(args.student_nfe), steps)
        
        for pos in ones_positions:
            ax.axvline(x=pos, ymax=1, linestyle='--', color='green', alpha=0.7)
            ax.scatter(pos, 1, color='red', s=50, zorder=10)
        
        ax.set_xticks(range(args.student_nfe))
        
        
    if logit_model.config.model_variant == 'constant':
        timesteps = timesteps[-1]
        fig, ax = plt.subplots(figsize=(6, 3))
        draw_plot(ax, timesteps)
    else:
        timesteps = timesteps[-args.logging.display_k:]
        fig, ax = plt.subplots(args.logging.display_k, 1, figsize=(6, 3 * args.logging.display_k))
    
        for k in range(args.logging.display_k):
            draw_plot(ax[k], timesteps[k])
    
    fig.suptitle(f"Timesteps, step {step}")
    plt.tight_layout()
    
    run['val/timesteps_plot'].append(fig, step=step)
    
    plt.close(fig)

    run[f'val/{args.metric}'].append(float(metric_value / count), step=step)


def log_test(
    pipe: BaseSDXL, 
    helper: CachingTimestepHelper, 
    logit_model: BaseLogitModel, 
    test_noise: torch.Tensor, 
    test_prompts: List[str], 
    epoch: int, 
    global_step: int, 
    args: Munch, 
    run: neptune.Run
):

    metric_names = ['LPIPS', 'PLPIPS', 'L1', 'AQ', 'IQ', 'HPS', 'CLIP', 'ImageReward']
    test_dataloader = DataLoader(test_prompts, batch_size=args.batch_size, shuffle=False)

    # TEACHER MODEL
    teacher_data_path = os.path.join(
        ROOT, 'DATA', 'cachers', 'eval_data', 
        f"teacher_test_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.test_size}.pt"
    )
    imgs_teacher = generate_data(
        pipe, helper, [], test_dataloader, test_noise, teacher_data_path, args, 
        args.teacher_nfe, "teacher test data", is_test=True
    )


    student_data_path = os.path.join(args.checkpoint_path, str(epoch), "images.pt")
    
    imgs_student = generate_data(
        pipe, helper, logit_model, test_dataloader, test_noise, student_data_path, args, 
        args.student_nfe, "student test data", is_test=True
    )

    def calc_metrics(metric_names, **data):
        '''calculate metrics by looping over all metrics in metrics_registry''' 
        res_metrics = {}
        for metric_name in metric_names:
            metricInstance = metric_registry[metric_name]()
            res_metrics[metric_name] = metricInstance(**data)

            gc.collect()
            torch.cuda.empty_cache()

            print(f'{metric_name}: {res_metrics[metric_name]}', flush=True)
        return res_metrics

    metrics = calc_metrics(
        metric_names=metric_names,
        imgs_gen=imgs_student,
        imgs_real=imgs_teacher,
        prompts=test_prompts,
        device=args.device,
    )
    run["metrics"].append(metrics, step=global_step)
    
    
def generate_data(pipe, helper, ts_to_skip, dataloader, noise_tensor, data_path, args, nfe, dataset_description="", is_test=False):
    # нам дают либо таймстепы сразу, либо модельку, их выдающую
    predict = not isinstance(ts_to_skip, list) 
    if predict:
        model = ts_to_skip
    
    if os.path.exists(data_path):
        generated = torch.load(data_path, weights_only=True)
        assert len(noise_tensor) == len(generated)
        print(f"\nFOUND READY {dataset_description}: {data_path}.\n")
    else:
        seed_everything()

        if not predict:
            not_cached = sorted(set(range(nfe)) - set(ts_to_skip))
            print(f"\n\tNOT CACHED: {not_cached}")
            print(f"\tCACHED: {ts_to_skip}")

        if not os.path.exists(data_path):

            outputs = []
            with torch.no_grad():
                for batch_idx, anns in enumerate(tqdm(dataloader, dataset_description, leave=False)):
                    batch_start = batch_idx * args.batch_size
                    batch_end = batch_start + args.batch_size
                    current_noise = noise_tensor[batch_start:batch_end]
                    
                    prompt_embeddings = pipe.encode_prompt(anns, device=args.device)
                    if predict:
                        logits = model(prompt_embeddings)
                        
                        mode_logits = sample_exp(logits, inference=True)
                        ts_to_skip = torch.argsort(mode_logits)[args.num_steps:] + 1
                        
                        not_cached = sorted(set(range(nfe)) - set(ts_to_skip))
                        print(f"\n\tNOT CACHED: {not_cached}")
                        print(f"\tCACHED: {ts_to_skip}")

                    with helper.inference(timesteps=ts_to_skip):
                        gen = pipe(
                            # anns,
                            prompt_embeddings=prompt_embeddings,
                            num_inference_steps=nfe,
                            guidance_scale=args.gs,
                            latents=current_noise,
                            output_type="pt"
                        )
                    outputs.append(gen)

            generated = torch.cat(outputs, dim=0).cpu()
            if is_test:
                generated = (generated + 1) * 0.5           # [-1,1] -> [0,1]
            
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            torch.save(generated, data_path)
            
            print(f"\tDATA CREATED AND SAVED TO: {data_path}")
    
    return generated


def generate_init_data(pipe, helper, train_dataloader, val_dataloader, train_noise, val_noise, args):
    
    # TEACHER TRAIN DATA
    teacher_train_data_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"teacher_train_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pt"
    )
    teacher_train_data = generate_data(
        pipe, helper, [], train_dataloader, train_noise, teacher_train_data_path, args,
        args.teacher_nfe, dataset_description="teacher train data", is_test=False
    )

    # TEACHER VAL DATA
    teacher_val_data_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"teacher_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pt"
    )
    teacher_val_data = generate_data(
        pipe, helper, [], val_dataloader, val_noise, teacher_val_data_path, args,
        args.teacher_nfe, dataset_description="teacher val data", is_test=False
    )

    # DEEPCACHE3 VAL DATA
    baseline_val_data_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"{args.logit_predictor.init_logits}_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pt"
    )
    if args.logit_predictor.init_logits == "deepcache-3":
        stride = 3
    elif args.logit_predictor.init_logits == "deepcache-4":
        stride = 4
    not_cached_steps = list(range(0, int(args.teacher_nfe), stride))
    ts_to_skip = sorted(set(list(range(args.student_nfe))) - set(not_cached_steps))
    baseline_val_data = generate_data(
        pipe, helper, ts_to_skip, val_dataloader, val_noise, baseline_val_data_path, args,
        args.student_nfe, dataset_description="DEEPCACHE3 val data", is_test=False
    )

    return teacher_train_data, teacher_val_data, baseline_val_data


def init_noises(args, latent_size, type, device):

    seed_everything()

    # TRAIN NOISE
    train_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"train_noise_{args.max_samples}.pt")
    if os.path.exists(train_noise_path):
        train_noise = torch.load(train_noise_path, weights_only=True, map_location='cpu').to(args.device)
    else:
        train_noise = torch.randn(
            (args.max_samples, 4, latent_size, latent_size), dtype=type, device=device)
        
        os.makedirs(os.path.dirname(train_noise_path), exist_ok=True)
        torch.save(
            train_noise, 
            train_noise_path
        )

    # VAL NOISE
    val_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"val_noise_{args.max_samples}.pt")
    if os.path.exists(val_noise_path):
        val_noise = torch.load(val_noise_path, weights_only=True, map_location='cpu').to(args.device)
    else:
        val_noise = torch.randn(
            (args.max_samples, 4, latent_size, latent_size), dtype=type, device=device)
        
        os.makedirs(os.path.dirname(val_noise_path), exist_ok=True)
        torch.save(
            val_noise, 
            val_noise_path
        )
    
    # TEST NOISE
    test_noise_path = os.path.join(ROOT, "DATA", "cachers", "noises", f"test_noise_{args.test_size}.pt")
    if os.path.exists(test_noise_path):
        test_noise = torch.load(test_noise_path, weights_only=True, map_location='cpu').to(args.device)
    else:
        test_noise = torch.randn(
            (args.test_size, 4, latent_size, latent_size), dtype=type, device=device)
        
        os.makedirs(os.path.dirname(test_noise_path), exist_ok=True)
        torch.save(
            test_noise, 
            test_noise_path
        )

    return train_noise, val_noise, test_noise


def reinforce_training_loop(
    pipe,
    train_dataloader,
    val_dataloader,
    test_prompts,
    metric_name,
    args,
    run
):
    
    seed_everything()
    
    helper = pipe.cacher
    latent_size = pipe.unet.config.sample_size
    train_noise, val_noise, test_noise = init_noises(args, latent_size, pipe.dtype, pipe.device)

    if metric_name.lower() == "patched-lpips":
        metric = PatchedLPIPS(device=pipe.device)

    args.num_steps = args.num_steps - 1
    
    logit_model = load_logit_model(
        args.student_nfe - 1, 
        args.logit_predictor,
        dtype=torch.float32, 
        device=args.device
    ) 
    
    logits_grad_mean = deque([np.nan] * 8, maxlen=8)
    model_grad_mean = deque([np.nan] * 8, maxlen=8) 
    
    optim = torch.optim.Adam(logit_model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
    alphas = torch.cat([
        torch.linspace(args.alpha, 0, args.warming_entropy_epochs),
        torch.zeros(args.epochs - args.warming_entropy_epochs)
    ])

    teacher_train_images, teacher_val_images, baseline_val_images = generate_init_data(
        pipe,
        helper,
        train_dataloader,
        val_dataloader,
        train_noise,
        val_noise,
        args
    )

    print("\nEverything ready for training!\n")

    
    for epoch in tqdm(range(args.epochs), 'Epochs'):

        if epoch % args.logging.eval_freq == 0:
            log_validation(
                pipe, helper, logit_model, val_noise, 
                val_dataloader, teacher_val_images, baseline_val_images,
                metric, epoch * len(train_dataloader), args, run
            )
        
        for batch_idx, anns in enumerate(tqdm(train_dataloader, 'Training', leave=False)):
            prompt_embeddings = pipe.encode_prompt(anns, device=args.device)
            logits = logit_model(prompt_embeddings)
            
            global_step = epoch * len(train_dataloader) + batch_idx    
            optim.zero_grad()
            
            batch_start = batch_idx * args.batch_size
            batch_end = batch_start + args.batch_size
            original = teacher_train_images[batch_start:batch_end].to(pipe.device)
            current_noise = train_noise[batch_start:batch_end]
                
            metrics = torch.empty((args.num_samples, args.batch_size), device=args.device)       # [num_samples, batch_size]
            logprobs = torch.zeros(args.num_samples, device=args.device)                         # [num_samples]
            
            for i in tqdm(range(args.num_samples), 'Reinforce steps', leave=False):
                perturbed_logits = sample_exp(logits, inference=False)
                
                ts, log_prob = top_k_log_prob(logits, perturbed_logits, args.num_steps)

                with helper.inference(timesteps=ts):
                    generated = pipe(
                        prompt_embeddings=prompt_embeddings,
                        num_inference_steps=args.student_nfe, 
                        guidance_scale=args.gs,
                        latents=current_noise,
                        output_type='pt'
                    )
                
                logprobs[i] = log_prob
                metrics[i, :] = metric.calculate(original, generated).detach()
            
            metrics_mean = metrics.mean(dim=0)                                                   # without regularization
            metrics = metrics + alphas[epoch] * logprobs[:, None]                                # add entropy regularizer

            metrics_mean_reg = metrics.mean(dim=0)                                                   # [batch_size]
            metrics_corrected = (metrics - metrics_mean_reg[None, :]).detach()                       # [num_samples, batch_size]
            
            loss = (metrics_corrected * logprobs[:, None]).mean() * (args.num_samples) / (args.num_samples - 1)
            loss.backward()
            optim.step()
            
            log_train(
                logit_model, loss, logprobs, metrics_mean.mean(), metrics_mean_reg.mean(), 
                model_grad_mean, logits_grad_mean, scheduler, global_step, args, run
            )

        scheduler.step()

        if (epoch + 1) % args.logging.test_freq == 0:
            log_test(
                pipe, helper, logit_model, test_noise, 
                test_prompts, epoch+1, global_step, args, run
            )

    log_validation(
        pipe, helper, logit_model, val_noise, 
        val_dataloader, teacher_val_images, baseline_val_images,
        metric, (epoch+1) * len(train_dataloader), args, run
    )
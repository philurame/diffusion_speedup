import torch
import numpy as np
import neptune
from neptune.types import File
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import gc
import sys
import pickle
from PIL import Image
from tqdm import tqdm
from munch import Munch
from collections import deque
from datetime import datetime
from typing import List
from torch_ema import ExponentialMovingAverage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from lib.models.SDXL.sdxl import BaseSDXL
from lib.models.models_utils.caching_timestep_helper import CachingTimestepHelper

from Training.cachers.logit_predictor import load_logit_model, BaseLogitModel
from Training.models import seed_everything
from Training.cachers.loss_cachers import PatchedLPIPS, HPSMetric, PLPIPS_HPS
from Training.cachers.reinforce_logic import sample_exp, top_k_log_prob
from registries import metric_registry

def save_pkl(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

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
        p for p in logit_model.model_parameters() 
        if p.grad is not None and p.requires_grad
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

    log_dict = {
        'train loss': loss.item(),
        f'train {args.metric} before baseline and regularization': before_baseline_and_reg.item(),
        f'train {args.metric} before baseline': before_baseline.item(),
        'log prob': logprobs.mean().item(),
        'logits grad norm': logits_grad_norm,
        'logits grad norm moving average': np.nanmean(logits_grad_mean),
        'model grad norm': model_norm,
        'model grad norm moving average': np.nanmean(model_grad_mean),
    }

    lr_list = scheduler.get_last_lr()
    if args.logit_predictor.model_variant == 'constant':
        log_dict['logits lr'] = lr_list[0]
        log_dict['model lr'] = None
    else:
        log_dict['logits lr'] = lr_list[0]
        log_dict['model lr'] = lr_list[1]
    
    run["train"].append(log_dict, step=step)
    
    
    # run["train"].append({
    #     'train loss': loss.item(),
    #     f'train {args.metric} before baseline and regularization': before_baseline_and_reg.item(),
    #     f'train {args.metric} before baseline': before_baseline.item(),
    #     'log prob': logprobs.mean().item(),
    #     'logits grad norm': logits_grad_norm,
    #     'logits grad norm moving average': np.nanmean(logits_grad_mean),
    #     'model grad norm': model_norm,
    #     'model grad norm moving average': np.nanmean(model_grad_mean),
    #     'model lr': scheduler.get_last_lr()[1], 
    #     'logits lr': scheduler.get_last_lr()[0]
    # }, step=step)


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
    run: neptune.Run,
    prefix: str = ''
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
    logits = []
    
    with torch.no_grad():
        for anns in tqdm(val_dataloader, 'Validation', leave=False):
            orig          = torch.stack([teacher_val_images[p]  for p in anns]).to(pipe.device)
            current_noise = torch.stack([val_noise[p]           for p in anns]).to(pipe.device)
            
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
            metric_value += metric.calculate(generated=gen, original=orig, prompts=anns).mean().item()
            count += 1
            
            original.append(orig.cpu())
            generated.append(gen.cpu())
            timesteps.append(ts_to_skip)
    
    logits = logits.detach().cpu()
    for i in range(len(logits)):
        run[f'val/logits/logit_{i+1}'].append(logits[i].item(), step=step)            

            
    original = torch.cat(original, dim=0)[-args.logging.display_k:]
    generated = torch.cat(generated, dim=0)[-args.logging.display_k:]
    baseline = torch.stack([baseline_imgs[p] for p in list(val_dataloader.dataset)])[-args.logging.display_k:]
    
    img = concat_images(original, generated, baseline)

    img = (img * 255).round().astype(np.uint8)

    def resize_image(img_array, target_size=(256, 256)):
        img = Image.fromarray(img_array)
        resized_img = img.resize(target_size, resample=Image.LANCZOS)
        return np.array(resized_img)

    img = resize_image(img, target_size=(512 * 3, args.logging.display_k * 512))

    run[f'val/{prefix}images'].append(
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
    
    run[f'val/{prefix}timesteps_plot'].append(fig, step=step)
    
    plt.close(fig)

    run[f'val/{prefix}{args.metric}'].append(float(metric_value / count), step=step)


def log_test(
    pipe: BaseSDXL, 
    helper: CachingTimestepHelper, 
    logit_model: BaseLogitModel, 
    test_noise: torch.Tensor, 
    test_prompts: List[str], 
    epoch: int, 
    global_step: int, 
    args: Munch, 
    run: neptune.Run,
    prefix: str = ''
):

    metric_names = ['LPIPS', 'PLPIPS', 'L1', 'AQ', 'IQ', 'HPS', 'CLIP', 'ImageReward']
    test_dataloader = DataLoader(test_prompts, batch_size=args.batch_size, shuffle=False)

    # TEACHER MODEL
    teacher_data_path = os.path.join(
        ROOT, 'DATA', 'cachers', 'eval_data', 
        f"teacher_test_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.test_size}.pt"
    )
    
    imgs_teacher = torch.stack(list(
        generate_data(
            pipe, helper, [], test_dataloader, test_noise, teacher_data_path, args, 
            args.teacher_nfe, "TEACHER TEST DATA", is_test=True
        ).values()
    ))

    # STUDENT MODEL
    student_data_path = os.path.join(args.checkpoint_path, str(epoch), f"{prefix}images.pt")
    imgs_student = torch.stack(list(
        generate_data(
            pipe, helper, logit_model, test_dataloader, test_noise, student_data_path, args, 
            args.student_nfe, "STUDENT TEST DATA", is_test=True
        ).values()
    ))

    def calc_metrics(metric_names, **data):
        '''calculate metrics by looping over all metrics in metrics_registry''' 
        res_metrics = {}
        for metric_name in metric_names:
            metricInstance = metric_registry[metric_name]()
            res_metrics[metric_name] = metricInstance(**data)

            del metricInstance
            gc.collect()
            torch.cuda.empty_cache()

            print(f'{metric_name}: {res_metrics[metric_name]}', flush=True)
        return res_metrics

    with torch.inference_mode(): 
        metrics = calc_metrics(
            metric_names=metric_names,
            imgs_gen=imgs_student,
            imgs_real=imgs_teacher,
            prompts=test_prompts,
            device=args.device,
        )
    run[f"{prefix}metrics"].append(metrics, step=global_step)

    del imgs_teacher, imgs_student
    gc.collect()
    torch.cuda.empty_cache()    
    
    
def generate_data(
    pipe, helper, ts_to_skip, dataloader, 
    noise_map, data_path, args, nfe, 
    dataset_description="", is_test=False
):
    
    if os.path.exists(data_path):
        images = load_pkl(data_path)
        print(f"\nFOUND READY {dataset_description}: {data_path}.\n")
    else:
        seed_everything(args.seed)

        # нам дают либо таймстепы сразу, либо модельку, их выдающую
        predict = not isinstance(ts_to_skip, list) 
        if predict:
            model = ts_to_skip

        images = {}
        with torch.inference_mode():
            for anns in tqdm(dataloader, dataset_description, leave=False):

                current_noise = torch.stack([noise_map[p] for p in anns]).to(pipe.device)
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
                        prompt_embeddings=prompt_embeddings,
                        num_inference_steps=nfe,
                        guidance_scale=args.gs,
                        latents=current_noise,
                        output_type="pt"
                    )

                for p, img in zip(anns, gen):
                    img = (img + 1) * 0.5 if is_test else img  # [-1,1] -> [0,1] for test data
                    images[p] = img.cpu()

                del gen
                gc.collect()
                torch.cuda.empty_cache()   

        save_pkl(images, data_path)
    return images


def generate_init_data(pipe, helper, train_dataloader, val_dataloader, train_noise_map, val_noise_map, args):
    
    # TEACHER TRAIN DATA
    teacher_train_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"teacher_train_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pkl"
    )
    teacher_train_data = generate_data(
        pipe, helper, [], train_dataloader, train_noise_map, 
        teacher_train_path, args, args.teacher_nfe, 
        dataset_description="TEACHER TRAIN DATA", is_test=False
    )

    # TEACHER VAL DATA
    teacher_val_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"teacher_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pkl"
    )
    teacher_val_data = generate_data(
        pipe, helper, [], val_dataloader, val_noise_map, 
        teacher_val_path, args, args.teacher_nfe, 
        dataset_description="TEACHER VAL DATA", is_test=False
    )

    # BASELINE VAL DATA (DEEPCACHE)
    baseline_val_data_path = os.path.join(
        ROOT, "DATA", "cachers", "train_data", 
        f"deepcache-{args.logit_predictor.baseline_stride}_val_{args.model_name}_{args.solver}_{args.scheduler}_{args.teacher_nfe}_{args.max_samples}.pkl"
    )
    
    stride = args.logit_predictor.baseline_stride
    not_cached_steps = list(range(0, int(args.teacher_nfe), stride))
    ts_to_skip = sorted(set(list(range(args.student_nfe))) - set(not_cached_steps))
    baseline_val_data = generate_data(
        pipe, helper, ts_to_skip, val_dataloader, val_noise_map, 
        baseline_val_data_path, args, args.student_nfe, 
        dataset_description="BASELINE VAL DATA", is_test=False
    )

    return teacher_train_data, teacher_val_data, baseline_val_data


def init_noises(args, latent_size, type, device, all_prompts):

    def build_or_load(prompts, name):
        path = os.path.join(ROOT, "DATA", "cachers", "noises", f"{name}.pkl")
        if os.path.exists(path):
            return load_pkl(path)
        m = {
            p: torch.randn((4, latent_size, latent_size), dtype=type, device='cpu') for p in prompts
        }
        save_pkl(m, path)
        return m

    seed_everything(args.seed)

    train_noise = build_or_load(all_prompts['train'], f"train_noise_{len(all_prompts['train'])}")
    val_noise   = build_or_load(all_prompts['val'],   f"val_noise_{len(all_prompts['val'])}")
    test_noise  = build_or_load(all_prompts['test'],  f"test_noise_{len(all_prompts['test'])}")

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
    
    seed_everything(args.seed)
    
    helper = pipe.cacher
    latent_size = pipe.unet.config.sample_size

    train_prompts = list(train_dataloader.dataset)
    val_prompts   = list(val_dataloader.dataset)
    all_prompts = {
        "train": train_prompts,
        "val":   val_prompts,
        "test":  test_prompts
    }

    train_noise, val_noise, test_noise = init_noises(args, latent_size, pipe.dtype, pipe.device, all_prompts)
    teacher_train_images, teacher_val_images, baseline_val_images = generate_init_data(
        pipe, helper, train_dataloader, val_dataloader,
        train_noise, val_noise, args
    )

    if metric_name.lower() == "patched-lpips":
        metric = PatchedLPIPS(device=pipe.device)
    elif metric_name.lower() == "hps":
        metric = HPSMetric(device=pipe.device)
    elif metric_name.lower() == "plpips_hps":
        metric = PLPIPS_HPS(device=pipe.device)

    args.num_steps = args.num_steps - 1
    
    logit_model = load_logit_model(
        args.student_nfe - 1, 
        args.logit_predictor,
        dtype=torch.float32, 
        device=args.device
    ) 
    
    logits_grad_mean = deque([np.nan] * 8, maxlen=8)
    model_grad_mean = deque([np.nan] * 8, maxlen=8) 
    
    # optim = torch.optim.Adam(logit_model.parameters(), args.lr)
    
    if args.logit_predictor.model_variant == 'constant':
        optim = torch.optim.Adam(logit_model.parameters(), args.logits_lr)
    else:
        optim = torch.optim.Adam(
            [
                {"params": logit_model.logits, "lr": args.logits_lr},
                {"params": logit_model.model_parameters(), "lr": args.model_lr} 
            ]
        )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
    alphas = torch.cat([
        torch.linspace(args.alpha, 0, args.warming_entropy_epochs),
        torch.zeros(args.epochs - args.warming_entropy_epochs)
    ])
    
    ema_enabled = False

    print("\nEverything ready for training!\n")
    
    for epoch in tqdm(range(args.epochs), 'Epochs'):
        if epoch == args.warming_entropy_epochs and args.logit_predictor.model_variant != 'constant':
            ema_enabled = True
            ema = ExponentialMovingAverage(
                logit_model.model_parameters(), decay=args.ema_decay
            )

        if epoch % args.logging.eval_freq == 0:
            log_validation(
                pipe, helper, logit_model, val_noise, 
                val_dataloader, teacher_val_images, baseline_val_images,
                metric, epoch * len(train_dataloader), args, run
            )
            
            if ema_enabled:
                with ema.average_parameters():
                    log_validation(
                        pipe, helper, logit_model, val_noise, 
                        val_dataloader, teacher_val_images, baseline_val_images,
                        metric, epoch * len(train_dataloader), args, run, prefix='ema_'
                    )
            
        for batch_idx, anns in enumerate(tqdm(train_dataloader, 'Training', leave=False)):
            prompt_embeddings = pipe.encode_prompt(anns, device=args.device)
            logits = logit_model(prompt_embeddings)
            
            global_step = epoch * len(train_dataloader) + batch_idx    
            optim.zero_grad()

            original      = torch.stack([teacher_train_images[p] for p in anns]).to(pipe.device)
            current_noise = torch.stack([train_noise[p]          for p in anns]).to(pipe.device)
                
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
                metrics[i, :] = metric.calculate(
                    generated=generated,
                    original=original,
                    prompts=anns
                ).detach()
            
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

            del metrics, logprobs
            gc.collect()
            torch.cuda.empty_cache()

        scheduler.step()

        if (epoch + 1) % args.logging.test_freq == 0 or (epoch + 1) == args.epochs:
            log_test(
                pipe, helper, logit_model, test_noise, 
                test_prompts, epoch+1, global_step, args, run
            )
            
            if ema_enabled:
                with ema.average_parameters():
                    log_test(
                        pipe, helper, logit_model, test_noise, 
                        test_prompts, epoch+1, global_step, args, run, prefix='ema_'
                    )
            
            saved = {
                "logit_model": logit_model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            if ema_enabled: 
                saved["ema"] = ema.state_dict()
            torch.save(saved, os.path.join(args.checkpoint_path, str(epoch+1), 'model.pt'))

    log_validation(
        pipe, helper, logit_model, val_noise, 
        val_dataloader, teacher_val_images, baseline_val_images,
        metric, (epoch+1) * len(train_dataloader), args, run
    )
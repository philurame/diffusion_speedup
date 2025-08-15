import os
import sys
import yaml
import click
import torch
from munch import Munch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from registries import import_dir, model_registry, solver_registry, scheduler_registry
import_dir(os.path.join(ROOT, 'lib'))
from lib.datasets.r_coco import COCO_SHORT


def parse_steps(spec: str):
    return [int(s.strip()) for s in spec.split(',') if s.strip() != ""]


def to_ts_skip(not_cached_zero, student_nfe):
    all_steps_zero = list(range(student_nfe))
    cached_zero = sorted(set(all_steps_zero) - set(not_cached_zero))
    cached_zero = [s for s in cached_zero if s != 0]  # первый шаг не кэшируем
    return torch.tensor([s + 1 for s in cached_zero], dtype=torch.long)  # 1-базовая индексация


@click.command()
@click.option("--config", type=str, required=True)
@click.option("--not-cached", "not_cached_spec", type=str, required=True)
@click.option("--device", type=str, default="cuda")
@click.option("--val_size", type=int, default=1000)
@click.option("--val_batch_size", type=int, default=8)
def main(config, not_cached_spec, device):
    with open(config, "r") as f:
        args = Munch(yaml.safe_load(f))
    args.device = device


    pipe = model_registry[args.model_name].from_pretrained(device=args.device)
    SolverClass = solver_registry[args.solver]
    SchedulerClass = scheduler_registry[args.scheduler]
    class SolverSchedulerConstructor(SchedulerClass, SolverClass): pass
    pipe.scheduler = SolverSchedulerConstructor(config=pipe.scheduler_config)
    helper = pipe.cacher


    coco = COCO_SHORT()
    val_dataloader = DataLoader(coco.prompts[-args.val_size:], batch_size=args.val_batch_size, shuffle=False)

    latent_size = pipe.unet.config.sample_size
    val_noise = torch.randn((args.val_batch_size, 4, latent_size, latent_size), dtype=pipe.dtype, device=args.device)

    not_cached_steps = parse_steps(not_cached_spec)
    ts_to_skip = to_ts_skip(not_cached_steps, args.student_nfe)

    outputs = []
    pipe.eval()
    with torch.no_grad():
        for anns in val_dataloader:
            with helper.inference(timesteps=ts_to_skip):
                gen = pipe(
                    anns,
                    num_inference_steps=args.student_nfe,
                    guidance_scale=args.gs,
                    latents=val_noise[:len(anns)],
                    output_type="pt"
                )
            outputs.append(gen)

    generated = torch.cat(outputs, dim=0).cpu()
    out_path = os.path.join(
        ROOT, "DATA",
        f"val_gen_{args.model_name}_{args.solver}_{args.scheduler}_nc{'-'.join(map(str, not_cached_steps))}_nfe{args.student_nfe}_v{args.val_size}.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(generated, out_path)
    print(out_path)


if __name__ == "__main__":
    main()

# python generate_val_min.py --config configs/my.yaml --not-cached "0,3,5" --device cuda
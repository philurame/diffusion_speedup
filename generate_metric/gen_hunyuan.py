# from setproctitle import setproctitle
# setproctitle("philurame short_evaluate.py") 

import sys, torch, tqdm, imageio, os, gc, json
ROOT = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/Training'
sys.path.append(ROOT)
from models import *

DEVICE = int(sys.argv[1])

with open('/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/hunyuan_prompts.txt', 'r') as f:
  prompts = f.readlines()

n_total = len(prompts)

SOLVER_SCHEDULER_NFE = [
  'FLOW2-LOG_FLOW_100',
  # 'FLOW2_FLOW_100',
  'FLOW_FLOW_100',
  # 'FLOW2-LOG_FLOW5_15',
  # 'FLOW2-LOG_FLOW9_15',
  
  # 'FLOW2-LOG_FLOW_20',
  # 'FLOW2_FLOW_20',
  # 'FLOW_FLOW_20',
  # 'FLOW2-LOG_FLOW5_20',
  # 'FLOW2-LOG_FLOW9_20',
]


print(DEVICE)

pipe = None
for comb in SOLVER_SCHEDULER_NFE:
  print(comb)
  solver, scheduler, nfe = comb.split('_')
  nfe = int(nfe)
  pipe = construct_pipeline(solver, scheduler, 'HUNYUAN_BASE', device=f'cuda:{DEVICE}', pipe=pipe)

  _id = f'{solver}_{scheduler}_{nfe}'
  video_p = f'/workspace-SR008.fs2/philurame/VMODEL/VIDEOS_HUNYUAN/{_id}'

  if not os.path.exists(video_p):
    os.makedirs(video_p, exist_ok=True)

  prompt_dict = {}

  for i in tqdm.tqdm(range(len(prompts))):
    if os.path.exists(f'{video_p}/{i}.mp4'): continue
    prompt = prompts[i]

    seed = i
    generator = torch.Generator(device='cpu').manual_seed(seed)
    video = pipe(prompt=prompt, num_inference_steps=nfe, generator=generator, height=640, width=640, output_type='video')

    vp = f'{video_p}/{i}.mp4'

    imageio.mimwrite(vp, 
      video.squeeze(),
      fps=18, 
      quality=10
    )
    prompt_dict[f'{video_p}/{i}.mp4'] = prompt

    torch.cuda.empty_cache()
    gc.collect()
  

  prompts_p = f'/workspace-SR008.fs2/philurame/VMODEL/PROMPTS_HUNYUAN/{_id}.json'
  with open(prompts_p, 'w') as f:
    json.dump(prompt_dict, f)
      
print('finished', flush=True)






import imageio, gc, json, os, torch, subprocess, tempfile
import torch.nn.functional as F

class ClassRegistry:
  def __init__(self):
    self._registry = {}
  def add_to_registry(self, name):
    def decorator(cls):
      self._registry[name] = cls
      return cls
    return decorator
  def __iter__(self):
    return iter(self._registry.keys())
  def __getitem__(self, name):
    return self._registry[name]
  def get(self, name, default=None):
    return self._registry.get(name, default)

metric_registry = ClassRegistry()

# ========================================================================================
# VBENCH
# ========================================================================================
@metric_registry.add_to_registry("VBENCH")
class VBENCH:
  def __init__(self, run_id):
    vdir_path    = '/workspace-SR008.fs2/philurame/VMODEL/VIDEOS_TRAIN'
    save_prompt_path = '/workspace-SR008.fs2/philurame/VMODEL/PROMPTS_TRAIN'
    prompts_path = '/workspace-SR008.fs2/philurame/DIFFUSION_SPEEDUP/DATA/hunyuan_prompts.txt'

    self.vbench_shell_path = "/workspace-SR008.fs2/philurame/scripts/help_vbench_train.sh"

    self.vbench_results_path = "/workspace-SR008.fs2/philurame/VMODEL/VBench/evaluation_results_train"

    with open(prompts_path, 'r') as f:
      self.prompts = f.readlines()

    self.prompts_path = os.path.join(save_prompt_path, f'{run_id}.json')
    self.save_path    = os.path.join(vdir_path, run_id)
    if not os.path.exists(self.save_path): os.makedirs(self.save_path, exist_ok=True)
    
    self.seed_shift = 0
    self.run_id = run_id
  
  def __call__(self, pipe, timesteps_model):
    prompt_dict = {}
    for i in range(len(self.prompts)):
      prompt = self.prompts[i]
      generator = torch.Generator(device='cpu').manual_seed(i+self.seed_shift)
      timesteps = timesteps_model.timesteps
      unet_timesteps = timesteps_model.unet_timesteps
      video = pipe(prompt=prompt, timesteps=timesteps, unet_timesteps=unet_timesteps, generator=generator, height=640, width=640, output_type='video')

      video_path = os.path.join(self.save_path, f'{i}.mp4')
      prompt_dict[video_path] = prompt
      imageio.mimwrite(
        video_path, 
        video.squeeze(),
        fps=18, quality=10
      )
    torch.cuda.empty_cache()
    gc.collect()

    with open(self.prompts_path, 'w') as f:
      json.dump(prompt_dict, f)
    
    # RUN VBENCH
    vbench_metrics = self.run_vbench(pipe.device)

    # COMPRESS VIDEOS
    for fname in os.listdir(self.save_path):
      if not fname.lower().endswith(".mp4"): continue
      vid_path = os.path.join(self.save_path, fname)
      self.compress_mp4_ffmpeg_inplace(vid_path)
    
    return vbench_metrics


  def run_vbench(self, device):
    device = str(torch.device(device).index)
    portn  = str(23031+int(device))

    cmd = [self.vbench_shell_path, device, self.run_id, portn]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print("stdout:", result.stdout)
    
    dimensions = ['imaging_quality', 'aesthetic_quality', 'motion_smoothness', 'dynamic_degree', 'overall_consistency', 'subject_consistency']
    vbench_metrics = {}
    for dimension in dimensions:
      result_path = os.path.join(self.vbench_results_path, dimension, self.run_id)
      res_file = [i for i in os.listdir(result_path) if 'eval_results' in i]
      if len(res_file) == 0:
        print(f"no results for {dimension}!!!!!")
        continue
      res_file = sorted(res_file, key=lambda x: os.path.getmtime(os.path.join(result_path, x)), reverse=True)[0]
      
      with open(os.path.join(result_path, res_file), 'r') as f:
        info = json.load(f)
      
      vbench_metrics[dimension] = list(info.values())[0][0]

    return vbench_metrics
  

  def compress_mp4_ffmpeg_inplace(self, path):
    dir_, name = os.path.split(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix=name + ".", dir=dir_)
    os.close(fd)

    cmd = [
      "ffmpeg", "-y",
      "-i", path,
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      "-movflags", "+faststart",
      tmp_path
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_path, path)
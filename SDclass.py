import os, gc, requests, random, shutil
import torch
import clip
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO
from pytorch_fid import fid_score
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from DeepCache import DeepCacheSDHelper
from tgate import TgateSDLoader, TgateSDXLLoader
from torch.profiler import profile, record_function, ProfilerActivity

class SDCompare:
  '''
  Class for gathering CLIP and FID statistics for Stable Diffusion 2.1
  depending on which scheduler, cache model and inference steps are used

  mscoco: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
  CLIP: https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
  FID: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
  '''

  # =============================================================================
  # Initialization
  # =============================================================================
  def __init__(self, scheduler_dict, cache_model="both", model='SD', clip_model='ViT-B/32', data_path='data'):
    '''
    Initializes Stable Diffusion pipeline with scheduler and cache model
    scheduler_dict is a dictionary with keys 'scheduler', 'params' and 'name'
    (scheduler_name, cache_model and model are also used for naming generated images folder)
    '''
    self.model = model
    self.cache_model = cache_model
    self.scheduler_dict = scheduler_dict
    self.clip_model = clip_model

    self.data_path = data_path
    os.makedirs(self.data_path, exist_ok=True)
    
    self.init_pipe()
    self.init_scheduler()
    self.init_cacher()
    self.init_COCO_data()
    self.init_CLIP_model()
    
    self.inference_steps = 15
  
  def init_pipe(self, model=None):
    '''
    Initializes Stable Diffusion pipeline
    '''
    model = model or self.model
    self.model = model

    if self.model == "SD":
      pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    elif self.model == "SDXL":
      pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    self.pipe = pipe.to("cuda")
    self.pipe.set_progress_bar_config(disable=True)

  def init_scheduler(self, scheduler_dict=None):
    '''
    Initializes scheduler
    '''
    scheduler_dict = scheduler_dict or self.scheduler_dict
    self.scheduler_dict = scheduler_dict

    self.pipe.scheduler = scheduler_dict['scheduler'].from_config(self.pipe.scheduler.config, **scheduler_dict.get('params', {}))

  def init_CLIP_model(self, clip_model=None):
    '''
    Initializes CLIP model
    '''
    clip_model = clip_model or self.clip_model
    self.clip_model = clip_model

    self.clip_model,  self.clip_preprocess = clip.load(clip_model)
    self.clip_model = self.clip_model.to("cuda").eval()
      
  def init_cacher(self, cache_model=None):
    '''
    Initializes cache model based on self.cache_model
    Possible values: "tgate", "deepcache", "both"
    '''
    cache_model = cache_model or self.cache_model
    self.cache_model = cache_model

    if self.cache_model in ["tgate", "both"]:
      if self.model == "SD":
        self.pipe = TgateSDLoader(self.pipe).to("cuda")
      elif self.model == "SDXL":
        self.pipe = TgateSDXLLoader(self.pipe).to("cuda")
    if self.cache_model in ["deepcache", "both"]:
      helper = DeepCacheSDHelper(pipe=self.pipe)
      helper.set_params(
          cache_interval=3,
          cache_branch_id=0,
      )
      helper.enable()
  
  def init_COCO_data(self, N_val=512, N_test=1024, path_coco_imgs='imgs_coco', path_coco_FID='imgs_coco_FID'):
    '''
    Downloads and extracts MSCOCO dataset with annotations and images
    Sets validation and test image ids
    '''
    path_coco_imgs = path_coco_imgs or os.join(self.data_path, 'imgs_coco')
    path_coco_FID = path_coco_FID or os.join(self.data_path, 'imgs_coco_FID')
    os.makedirs(path_coco_imgs, exist_ok=True)
    os.makedirs(path_coco_FID, exist_ok=True)

    if not os.path.exists(os.join(self.data_path, 'annotations')):
      annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
      annotations_path = 'annotations_trainval2017.zip'
      response = requests.get(annotations_url)
      open(annotations_path, 'wb').write(response.content)
      import zipfile
      with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
        zip_ref.extractall(self.data_path)
      os.remove(annotations_path)
    
    
    self.coco_imgs = COCO(os.join(self.data_path, 'annotations/instances_train2017.json'))
    self.coco_prompts = COCO(os.join(self.data_path, 'annotations/captions_train2017.json'))
    img_ids = self.coco_imgs.getImgIds()

    random.seed(42)
    random.shuffle(img_ids)
    self.img_ids = {'val': img_ids[0:N_val], 'test': img_ids[-N_test:]}

    # download images
    print('downloading images...')
    self.path_coco_imgs = path_coco_imgs
    already_downloaded = os.listdir(path_coco_imgs)
    images = self.coco_imgs.loadImgs(self.img_ids['val'] + self.img_ids['test'])
    for img in tqdm(images):
      if f"{img['id']}.png" in already_downloaded:
        continue
      img_url = img['coco_url']
      img_data = requests.get(img_url).content
      
      with open(f"{path_coco_imgs}/{img['id']}.png", 'wb') as handler:
        handler.write(img_data)
    
    # copy and resize images to 299x299
    already_downloaded = os.listdir(path_coco_FID)
    for img_id in tqdm(self.img_ids['val'] + self.img_ids['test']):
      img_coco = Image.open(f"{path_coco_imgs}/{img_id}.png")
      img_coco = img_coco.resize((299, 299), Image.ANTIALIAS)
      img_coco.save(f"{path_coco_FID}/{img_id}.png")

  
  # =============================================================================
  # __call__ and Utilities
  # =============================================================================
  def __call__(self, prompt, **kwargs):
    '''
    Returns generated image by prompt based on which cache model is used
    '''
    call_params = dict(
        prompt = prompt,
        num_inference_steps = self.inference_steps,
    )

    if self.cache_model == "deepcache":
      call_params.update(kwargs)
      return self.pipe(**call_params).images[0]

    call_params['gate_step'] = max(call_params['num_inference_steps']//2.5, 1)
    call_params.update(kwargs)
    return self.pipe.tgate(**call_params).images[0]
  
  def get_tflops(self, prompt, print_table=False, **kwargs):
    '''
    Returns GFLOPS depending on the cache_model, scheduler, inference_steps
    '''
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
      with record_function("model_inference"):
        self(prompt, **kwargs)

    if print_table:
      print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return round(prof.key_averages().flops/1e12,3)

  def get_clip_score(self, image, caption):
    '''
    Returns CLIP score for one image and one caption
    see https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
    '''
    image_input = self.clip_preprocess(image).unsqueeze(0).to("cuda")
    text_input  = clip.tokenize([caption]).to("cuda")

    with torch.no_grad():
      image_features = self.clip_model.encode_image(image_input)
      text_features  = self.clip_model.encode_text(text_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features / text_features.norm(dim=-1, keepdim=True)

    clip_score = torch.matmul(image_features, text_features.T).item()
    return clip_score
  

  # =============================================================================
  # CLIP and FID
  # =============================================================================
  def get_stats(self, path_gen=None, val_test='test', delete_gen_after=True):
    '''
    Generates conditional images and calculates CLIP on MSCOCO dataset
    Generates unconditional small images and calculates FID on resized MSCOCO dataset
    Returns stats dictionary for CLIP and FID
    '''

    if path_gen==None: 
      path_gen = f'imgs_{self.model}/cache_{self.cache_model}/{self.scheduler_dict["name"]}/{self.inference_steps}'
    self.path_gen = path_gen
    self.path_gen_FID  = os.path.join(path_gen,  'FID')
    os.makedirs(path_gen, exist_ok=True)
    os.makedirs(self.path_gen_FID,  exist_ok=True)

    stats = {}
    clip_scores = torch.zeros((len(self.img_ids[val_test]), 2), dtype=torch.float32)

    # CLIP loop:
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="CLIP")):
      torch.manual_seed(n)
      random.seed(n)

      ann_ids = self.coco_prompts.getAnnIds(imgIds=img_id)
      prompts = self.coco_prompts.loadAnns(ann_ids)
      prompt = random.choice([ann['caption'] for ann in prompts])

      img_gen_cond = self(prompt)
      img_coco = Image.open(f"{self.path_coco_imgs}/{img_id}.png")

      clip_gen  = float(self.get_clip_score(img_gen_cond, prompt))
      clip_real = float(self.get_clip_score(img_coco, prompt))
      clip_scores[n] = torch.tensor([clip_gen, clip_real])

    # CLIP stats
    stats['CLIP_mean'] = clip_scores[:,0].mean()
    stats['CLIP_diff'] = (clip_scores[:,0]-clip_scores[:,1]).abs().mean()
    print(f"CLIP mean: {stats['CLIP_mean']:.3f}, diff: {stats['CLIP_diff']:.3f}")


    gc.collect()
    torch.cuda.empty_cache()
    # FID loop:
    for n, img_id in enumerate(tqdm(self.img_ids[val_test], desc="FID")):
      torch.manual_seed(n)
      img_gen_uncond = self("")
      img_gen_uncond = img_gen_uncond.resize((299, 299), Image.ANTIALIAS)
      img_gen_uncond.save(f"{self.path_gen_FID}/{img_id}.png")
    
    # FID stat
    fid_value = fid_score.calculate_fid_given_paths([self.path_coco_FID, path_gen], batch_size=50, device='cuda', dims=2048)
    stats['FID'] = fid_value
    print(f"FID: {stats['FID']:.3f}")

    if delete_gen_after:
      shutil.rmtree(self.path_gen_FID)

    return stats
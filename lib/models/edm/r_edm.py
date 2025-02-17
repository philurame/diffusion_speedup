from lib.registries import model_registry
from lib.models.edm.dnnlib.util import open_url
import torch, os, pickle, sys

EDM_DIR = os.path.dirname(os.path.abspath(__file__))
if EDM_DIR not in sys.path:
  sys.path.insert(0, EDM_DIR)

@model_registry.add_to_registry('EDM')
class EDM:
  def __init__(self, **kwargs):
    self.unet = self._load_edm_net()
    self.scheduler_config = {
      "num_train_timesteps": 1000,
      "beta_start": 0.0001,
      "beta_end": 0.02,
      "beta_schedule": "linear",
    }
    self.device = kwargs.get('device', 'cuda')
    self.is_train = kwargs.get('is_train', False)

  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = cls(*args, **kwargs)
    return pipe
  
  def __call__(self, *args, **kwds):
    if self.is_train:
      return self._call_impl(*args, **kwds) 
    with torch.no_grad():
      return self._call_impl(*args, **kwds)
  
  def _call_impl(
    self,
    num_inference_steps=None,
    timesteps=None,
    generator=torch.Generator(),
    init_noise=None,
    **kwargs
    ):  

    device = self.device

    # initialize x_T batch
    image = init_noise or self._initialize_batch(generator)

    self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    for t in self.scheduler.timesteps:
        
        # predict x0
        sigma_t = self.scheduler.sigmas[self.scheduler.step_index].to(device)
        alpha_t = self.scheduler.sigma_to_alpha_t(sigma_t)
        x_0 = self.unet(image / alpha_t, sigma_t)

        # convert to eps
        model_output = (image - x_0 * alpha_t) / (alpha_t * sigma_t)
        image = self.scheduler.step(model_output, t, image)    
  
    return image
  
  def to(self, device):
    self.device = device
    self.unet = self.unet.to(device)
    return self
  
  def _initialize_batch(self, generator):
    if isinstance(generator, torch.Generator):
      generator = [generator]
    tensor_list = [torch.randn(3, 32, 32, generator=gen) for gen in generator]
    return torch.stack(tensor_list, dim=0).to(self.device)
  
  def _load_edm_net(self, local_path=None):
    if local_path is None:
      CURR_PATH = os.path.dirname(os.path.abspath(__file__))
      local_path = os.path.join(CURR_PATH, "edm-cifar10-32x32-uncond-vp.pkl")
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
    if os.path.exists(local_path):
      with open(local_path, 'rb') as f:
        net = pickle.load(f)['ema']
    else:
      with open_url(network_pkl, verbose=1) as f:
        net = pickle.load(f)['ema']

    for param in net.parameters():
      param.requires_grad = False

    net.eval()
    if torch.cuda.is_available():
      net.cuda()
    return net
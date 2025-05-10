import torch
import dnnlib
from .R3GAN.Networks import Discriminator
import pickle
import dnnlib

url_dict = {
  "ffhq64_d": "https://huggingface.co/brownvc/R3GAN-FFHQ-64x64/resolve/main/network-snapshot-final.pkl"
}
class _TFNetworkStub(dnnlib.EasyDict):
  pass

class _LegacyUnpickler(pickle.Unpickler):
  def find_class(self, module, name):
    if module == 'dnnlib.tflib.network' and name == 'Network':
      return _TFNetworkStub
    if module == 'training.networks_baseline':
      module = 'training.networks'
    if module[:12] == 'BaselineGAN.':
      module = 'R3GAN.' + module[12:]
    return super().find_class(module, name)

def load_network_pkl(f, force_fp16=False):
  data = _LegacyUnpickler(f).load()

  # Add missing fields.
  if 'training_set_kwargs' not in data:
    data['training_set_kwargs'] = None
  if 'augment_pipe' not in data:
    data['augment_pipe'] = None

  # Validate contents.
  assert isinstance(data['G'], torch.nn.Module)
  assert isinstance(data['D'], torch.nn.Module)
  assert isinstance(data['G_ema'], torch.nn.Module)
  assert isinstance(data['training_set_kwargs'], (dict, type(None)))
  assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

  return data

def load_r3gan_disc(local_path='', url_type="ffhq64_d"):
  if url_type == "None" and not local_path:
    return load_defualt_disc()
  
  if local_path:
    with open(local_path, 'rb') as f:
      d = load_network_pkl(f)
  else:
    url = url_dict[url_type]
    with dnnlib.util.open_url(url) as f:
      d = load_network_pkl(f)
  D = d['D']
  D.eval()
  return D

def load_defualt_disc():
  WidthPerStage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512]]
  BlocksPerStage = [2 * x for x in [1, 1, 1, 1, 1]]
  CardinalityPerStage = [3 * x for x in [32, 32, 32, 32, 16]]

  D = Discriminator(
    WidthPerStage=[*reversed(WidthPerStage)], 
    CardinalityPerStage=[*reversed(CardinalityPerStage)],
    BlocksPerStage=[*reversed(BlocksPerStage)], 
    ExpansionFactor=2,
  )
  return D


class DistAdversarialTraining:
  def __init__(self, local_path, lr, device):
    self.device = device
    
    self.Discriminator = load_r3gan_disc(
      local_path=local_path, url_type='None'
    )

    self.Discriminator.to(self.device)
    self.Opt = torch.optim.Adam(self.Discriminator.parameters(), lr=lr)


  def ZeroCenteredGradientPenalty(self, Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3])
      
  def AccumulateGeneratorGradients(self, FakeSamples, RealSamples, Conditions=None, Scale=1):
    self.Discriminator.eval()
    RealSamples = RealSamples.detach()
    
    FakeLogits = self.Discriminator(FakeSamples, Conditions)
    RealLogits = self.Discriminator(RealSamples, Conditions)
    
    RelativisticLogits = FakeLogits - RealLogits
    AdversarialLoss = torch.nn.functional.softplus(-RelativisticLogits)
    
    return (Scale * AdversarialLoss), [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
  
  def AccumulateDiscriminatorGradients(self, FakeSamples, RealSamples, Conditions=None, Gamma=0.2, Scale=1, is_train=True):
    if is_train: 
      self.Discriminator.train()
      RealSamples = RealSamples.detach().requires_grad_(True)
      FakeSamples = FakeSamples.detach().requires_grad_(True)
    else:
      self.Discriminator.eval()
    
    
    RealLogits = self.Discriminator(RealSamples, Conditions)
    FakeLogits = self.Discriminator(FakeSamples, Conditions)
    
    if is_train:
      R1Penalty = self.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
      R2Penalty = self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
    else:
      R1Penalty = torch.zeros_like(RealLogits)
      R2Penalty = torch.zeros_like(RealLogits)
    RelativisticLogits = RealLogits - FakeLogits
    AdversarialLoss = torch.nn.functional.softplus(-RelativisticLogits)
    
    DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
    
    return Scale * DiscriminatorLoss, [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
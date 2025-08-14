import pickle, os, torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
  def __init__(self, split, config, transfer_device):
    assert split in ('train', 'val'), "split must be 'train' or 'val'"

    dtype = torch.float16 if 'BASE' in config.model else torch.float32
    model_name   = config.model.split('_')[0]
    teacher_name = config.solver.teacher

    teacher_filename = f"{model_name}_{teacher_name}.pkl"
    with open(os.path.join(config.dataset['path'], teacher_filename), 'rb') as f: 
      data = pickle.load(f)[split]

    size = config.dataset[f'{split}_size']
    self.data = {}
    self.data['prompts'] = data['prompts'][:size]
    self.data['noise']   = data['noise'][:size].to(dtype)
    self.data['latents'] = data['latents'][:size]
    self.data['imgs']    = data['imgs'][:size]

    # preprocess imgs to be in [0,1]
    if self.data['imgs'].dtype == torch.uint8:
      self.data['imgs'] = self.data['imgs'].float()/255
    if self.data['imgs'].min() < -1.1: #[-1, 1]
      self.data['imgs'] = self.data['imgs']/2 + 0.5
    assert self.data['imgs'].max() < 2

    self.transfer_device = transfer_device

  def __len__(self):
    return len(self.data['prompts'])

  def __getitem__(self, idx):
    # handle slice
    if isinstance(idx, slice):
      start, stop, step = idx.indices(len(self))
      return {
        'prompts': self.data['prompts'][start:stop:step],
        'noise':   self.data['noise'][start:stop:step].to(self.transfer_device),
        'latents': self.data['latents'][start:stop:step].to(self.transfer_device),
        'imgs':    self.data['imgs'][start:stop:step].to(self.transfer_device),
      }

    # integer index
    if idx < 0: idx += len(self)
    if idx < 0 or idx >= len(self): raise IndexError('Index out of range')
    return {
      'prompt':  self.data['prompts'][idx],
      'noise':   self.data['noise'][idx].to(self.transfer_device),
      'latents': self.data['latents'][idx].to(self.transfer_device),
      'imgs':    self.data['imgs'][idx].to(self.transfer_device),
    }
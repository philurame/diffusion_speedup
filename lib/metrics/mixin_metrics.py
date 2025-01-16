class MetricDataMixin:
  def __init__(self, path_ddim=None, fake_imgs=None, real_imgs=None, real_anns=None,
               pipe=None, nfe=None, verbose=True, batch_size=512, tflops_prompt=None):
    self.path_ddim = path_ddim
    self.fake_imgs = fake_imgs
    self.real_imgs = real_imgs
    self.real_anns = real_anns
    self.pipe = pipe
    self.nfe = nfe
    self.verbose = verbose
    self.batch_size = batch_size
    self.tflops_prompt = tflops_prompt or 'a photograph of an astronaut riding a horse'
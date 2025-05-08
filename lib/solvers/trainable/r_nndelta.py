from lib.registries import solver_registry
from lib.solvers.trainable.set_solver import SETSOLVER
import torch
import torch.nn as nn

@solver_registry.add_to_registry("DeltaU")
class DeltaU(SETSOLVER):
  order = 3
  is_trainable = True
  def step(self, model_output, sample=None, **kwargs):
    self.model_outputs = [torch.zeros_like(model_output), torch.zeros_like(model_output)] + [i for i in self.model_outputs if i is not None]
    self.model_outputs = [sample] + self.model_outputs[-2:] + [model_output]

    input_tensor = torch.cat(self.model_outputs, dim=1)
    delta = self.nndelta(input_tensor)

    deis_coeffs = self.deis_coeffs[self.step_index]
    prev_sample = sum([i * j for i, j in zip(deis_coeffs, self.model_outputs)]) + delta

    self.step_index += 1
    return prev_sample + delta

  def set_train_solver(self, num_inference_steps, device=None):
    self.set_deis_coeffs(num_inference_steps)
    self.nndelta = UNet(in_channels=16, out_channels=4, base_filters=8).to(device)
    self.train_params = self.nndelta.parameters()



class DoubleConv(nn.Module):
  """A helper module that performs two successive convolutions with ReLU."""
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )
  def forward(self, x):
    return self.conv(x)

class UNet(nn.Module):
  def __init__(self, in_channels=16, out_channels=4, base_filters=64):
    """
    in_channels = 16 (4 for xᵗ + 4 each for ε₍ᵗ₎, ε₍ᵗ₊₁₎, ε₍ᵗ₊₂₎)
    out_channels = 4 (the x₍ᵗ₋₁₎ to predict)
    base_filters = 64 is an arbitrary choice for feature maps to start with.
    """
    super(UNet, self).__init__()

    # Encoder
    self.conv_down1 = DoubleConv(in_channels, base_filters)
    self.conv_down2 = DoubleConv(base_filters, base_filters * 2)
    self.conv_down3 = DoubleConv(base_filters * 2, base_filters * 4)
    self.pool = nn.MaxPool2d(2)

    # Bottleneck
    self.bottleneck = DoubleConv(base_filters * 4, base_filters * 8)

    # Decoder
    self.up1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
    self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
    self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
    self.conv_up1 = DoubleConv(base_filters * 8, base_filters * 4)
    self.conv_up2 = DoubleConv(base_filters * 4, base_filters * 2)
    self.conv_up3 = DoubleConv(base_filters * 2, base_filters)

    # Final output
    self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    # Initialize with random weights:
    torch.manual_seed(0)
    for m in self.modules():
      if hasattr(m, 'weight') and m.weight is not None:
        m.weight.data.uniform_(-1e-5, 1e-5)
      if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.uniform_(-1e-5, 1e-5)
      

  def forward(self, x):
    # Downsampling
    x1 = self.conv_down1(x)
    x2 = self.pool(x1)
    x2 = self.conv_down2(x2)
    x3 = self.pool(x2)
    x3 = self.conv_down3(x3)

    # Bottleneck
    x4 = self.pool(x3)
    x4 = self.bottleneck(x4)

    # Upsampling + skipping
    x5 = self.up1(x4)
    x5 = torch.cat([x5, x3], dim=1)
    x5 = self.conv_up1(x5)

    x6 = self.up2(x5)
    x6 = torch.cat([x6, x2], dim=1)
    x6 = self.conv_up2(x6)

    x7 = self.up3(x6)
    x7 = torch.cat([x7, x1], dim=1)
    x7 = self.conv_up3(x7)

    # Output
    out = self.output_conv(x7)
    return out
from lib.registries import solver_registry
import torch

@solver_registry.add_to_registry("UNIPC")
class UNIPC:
  order = 3
  is_trainable = False
  lower_order_final = True
  solver_type = "bh2"

  def step(self, model_output, sample=None, **kwargs):

    model_output_convert = self.convert_model_output(model_output, sample=sample)

    if self.step_index == 0: self.this_order = None

    if self.step_index > 0:
      sample = self.multistep_uni_c_bh_update(
        this_model_output=model_output_convert,
        last_sample=self.last_sample,
        this_sample=sample,
        order=self.this_order,
      )
    self.last_sample = sample

    self.model_outputs.append(model_output_convert)

    if self.lower_order_final:
      self.this_order = min(self.order, len(self.timesteps) - self.step_index)
    else:
      self.this_order = self.order
    self.this_order = min(self.this_order, self.step_index + 1)
    
    prev_sample = self.multistep_uni_p_bh_update(
      model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
      sample=sample,
      order=self.this_order,
    )

    print('O', prev_sample.sum())

    self.step_index += 1
    return prev_sample
  
  def convert_model_output(self, model_output, sample):
    sigma = self.sigmas[self.step_index]
    alpha_t = self.sigma_to_alpha_t(sigma)
    x0_pred = sample / alpha_t - sigma * model_output
    return x0_pred

  def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        order: int = None,
        **kwargs,
    ):
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t  = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        
        print(order, lambda_s0, h)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

  def multistep_uni_c_bh_update(
      self,
      this_model_output: torch.Tensor,
      *args,
      last_sample: torch.Tensor = None,
      this_sample: torch.Tensor = None,
      order: int = None,
      **kwargs,
  ) -> torch.Tensor:
      model_output_list = self.model_outputs

      m0 = model_output_list[-1]
      x = last_sample
      x_t = this_sample
      model_t = this_model_output

      sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
      alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
      alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

      lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
      lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

      h = lambda_t - lambda_s0
      device = this_sample.device

      rks = []
      D1s = []
      for i in range(1, order):
          si = self.step_index - (i + 1)
          mi = model_output_list[-(i + 1)]
          alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
          lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
          rk = (lambda_si - lambda_s0) / h
          rks.append(rk)
          D1s.append((mi - m0) / rk)

      rks.append(1.0)
      rks = torch.tensor(rks, device=device)

      R = []
      b = []

      hh = -h
      h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
      h_phi_k = h_phi_1 / hh - 1

      factorial_i = 1

      if self.solver_type == "bh1":
          B_h = hh
      elif self.solver_type == "bh2":
          B_h = torch.expm1(hh)
      else:
          raise NotImplementedError()

      for i in range(1, order + 1):
          R.append(torch.pow(rks, i - 1))
          b.append(h_phi_k * factorial_i / B_h)
          factorial_i *= i + 1
          h_phi_k = h_phi_k / hh - 1 / factorial_i

      R = torch.stack(R)
      b = torch.tensor(b, device=device)

      if len(D1s) > 0:
          D1s = torch.stack(D1s, dim=1)
      else:
          D1s = None

      # for order 1, we use a simplified version
      if order == 1:
          rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
      else:
          rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      if D1s is not None:
          corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
      else:
          corr_res = 0
      D1_t = model_t - m0
      x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)

      x_t = x_t.to(x.dtype)
      return x_t


@solver_registry.add_to_registry("UNIPC2")
class UNIPC2(UNIPC):
  order = 2
  is_trainable = False
  lower_order_final = True
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC3_")
class UNIPC3_(UNIPC):
  order = 3
  is_trainable = False
  lower_order_final = True
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC2_")
class UNIPC2_(UNIPC):
  order = 2
  is_trainable = False
  lower_order_final = True
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC3H")
class UNIPC3H(UNIPC):
  order = 3
  is_trainable = False
  lower_order_final = False
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC2H")
class UNIPC2H(UNIPC):
  order = 2
  is_trainable = False
  lower_order_final = False
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC3H_")
class UNIPC3H_(UNIPC):
  order = 3
  is_trainable = False
  lower_order_final = False
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC2H_")
class UNIPC2H_(UNIPC):
  order = 2
  is_trainable = False
  lower_order_final = False
  solver_type = "bh1"
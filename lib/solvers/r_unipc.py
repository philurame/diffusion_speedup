from lib.registries import solver_registry
import torch

@solver_registry.add_to_registry("UNIPC3")
class UNIPC3:
    order = 3
    is_trainable = False
    lower_order_final = True
    solver_type = "bh2"

    def step(self, model_output, sample=None, **kwargs):
        if kwargs.get("prediction_type", "epsilon") == "v_prediction":
          sigma_t = self.sigmas[self.step_index]
          alpha_t = self.sigma_to_alpha_t(sigma_t)
          model_output = alpha_t * (sample * sigma_t + model_output)
    
        # 1. Convert model_output once
        x0_pred = self._convert_model_output(model_output, sample)

        # 2. Conditional corrector update if not the first step
        if self.step_index > 0:
            sample = self._c_update(x0_pred)
          
        # 3. Record state and advance order
        self.last_sample = sample
        self.model_outputs.append(x0_pred)
        if self.lower_order_final:
            self.this_order = min(self.order, len(self.timesteps) - self.step_index)
        else:
            self.this_order = self.order
        self.this_order = min(self.this_order, self.step_index + 1)

        # 4. Increment step and perform predictor update
        prev = self._p_update(model_output, sample)

        self.step_index += 1
        return prev

    def _convert_model_output(self, model_output, sample):
        # Helper: compute x0_pred
        sigma = self.sigmas[self.step_index]
        alpha_t = self.sigma_to_alpha_t(sigma)
        return sample / alpha_t - sigma * model_output

    def _prep_coeffs(self, is_predictor):
        # 2. Common phi, h, and B_h computations
        sigma_t, sigma_s0 = self.sigmas[self.step_index+is_predictor], self.sigmas[self.step_index-1+is_predictor]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lam_t = torch.log(alpha_t) - torch.log(sigma_t)
        lam_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lam_t - lam_s0
        hh = -h
        phi1 = torch.expm1(hh)
        B_h = hh if self.solver_type == "bh1" else phi1
        return alpha_t, sigma_t, alpha_s0, sigma_s0, lam_s0, h, hh, phi1, B_h

    def _build_rks_D1s(self, order, lam_s0, h, is_predictor):
        # 3. Build rks and D1s lists
        m0 = self.model_outputs[-1]
        device = m0.device
        rks, D1s = [], []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[self.step_index-i-1+is_predictor])
            lam_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lam_si - lam_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks = torch.tensor(rks, device=device)
        return rks, torch.stack(D1s, dim=1) if D1s else None

    def _build_Rb(self, rks, phi1, hh, B_h, order):
        # 4. Build Vandermonde matrix R and vector b
        device = rks.device
        R, b = [], []
        factorial, phi_k = 1, phi1 / hh - 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(phi_k * factorial / B_h)
            factorial *= i + 1
            phi_k = phi_k / hh - 1 / factorial
        return torch.stack(R), torch.tensor(b, device=device)

    def _p_update(self, model_output, x):
      # Predictor (multistep uni-p) update
      # 5. Prepare coeffs for p: next timestep -> current
      alpha_t, sigma_t, alpha_s0, sigma_s0, lam_s0, h, hh, phi1, B_h = self._prep_coeffs(True)
      rks, D1s = self._build_rks_D1s(self.this_order, lam_s0, h, True)
      # Solve for rhos_p
      if D1s is not None and self.this_order > 2:
          R, b = self._build_Rb(rks, phi1, hh, B_h, self.this_order)
          rhos = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(x.dtype)
      else:
          rhos = torch.tensor([0.5], dtype=x.dtype, device=x.device)
      # Compute base x_t and add model corrections
      x_base = sigma_t / sigma_s0 * x - alpha_t * phi1 * self.model_outputs[-1]
      pred = torch.einsum("k,bkc...->bc...", rhos, D1s) if D1s is not None else 0
      x_t = (x_base - alpha_t * B_h * pred).to(x.dtype)
      return x_t

    def _c_update(self, m0_pred):
      x = self.last_sample
      # Corrector (multistep uni-c) update
      # 6. Prepare coeffs for c: current -> previous
      alpha_t, sigma_t, alpha_s0, sigma_s0, lam_s0, h, hh, phi1, B_h = self._prep_coeffs(False)
      rks, D1s = self._build_rks_D1s(self.this_order, lam_s0, h, False)
      # Solve for rhos_c
      if self.this_order == 1:
          rhos = torch.tensor([0.5], dtype=x.dtype, device=x.device)
      else:
          R, b = self._build_Rb(rks, phi1, hh, B_h, self.this_order)
          rhos = torch.linalg.solve(R, b).to(x.dtype)

      # Compute base x_t and apply corrections
      x_base = sigma_t / sigma_s0 * x - alpha_t * phi1 * self.model_outputs[-1]
      corr = torch.einsum("k,bkc...->bc...", rhos[:-1], D1s) if D1s is not None else 0
      D1_t = m0_pred - self.model_outputs[-1]
      x_t = (x_base - alpha_t * B_h * (corr + rhos[-1] * D1_t)).to(x.dtype)
      return x_t
  

@solver_registry.add_to_registry("UNIPC2")
class UNIPC2(UNIPC3):
  order = 2
  is_trainable = False
  lower_order_final = True
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC3_")
class UNIPC3_(UNIPC3):
  order = 3
  is_trainable = False
  lower_order_final = True
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC2_")
class UNIPC2_(UNIPC3):
  order = 2
  is_trainable = False
  lower_order_final = True
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC3H")
class UNIPC3H(UNIPC3):
  order = 3
  is_trainable = False
  lower_order_final = False
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC2H")
class UNIPC2H(UNIPC3):
  order = 2
  is_trainable = False
  lower_order_final = False
  solver_type = "bh2"

@solver_registry.add_to_registry("UNIPC3H_")
class UNIPC3H_(UNIPC3):
  order = 3
  is_trainable = False
  lower_order_final = False
  solver_type = "bh1"

@solver_registry.add_to_registry("UNIPC2H_")
class UNIPC2H_(UNIPC3):
  order = 2
  is_trainable = False
  lower_order_final = False
  solver_type = "bh1"
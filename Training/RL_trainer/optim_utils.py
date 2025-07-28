import torch
import torch.optim as optim

def get_optimizer(config, timesteps_model, pipe):
  train_params = []
  n_params_train = 0

  if config.timesteps.train:
    train_params.append({
      "params": [timesteps_model.timesteps_logits, timesteps_model.unet_timesteps_logits], 
      "lr": config.timesteps.lr
    })
    
    n_params_train += timesteps_model.timesteps_logits.numel()
    n_params_train += timesteps_model.unet_timesteps_logits.numel()
  
  if config.solver.train:
    pipe.scheduler.set_train_solver(timesteps=timesteps_model.timesteps, device=config.device)
    train_params.append({"params": pipe.scheduler.train_params, "lr": config.solver.lr})

    n_params_train += sum(p.numel() for p in pipe.scheduler.train_params)
  
  init_tensor = -1.*torch.ones(n_params_train, dtype=torch.float32, device=config.device)
  if config.solver.train:
    init_tensor[-sum(p.numel() for p in pipe.scheduler.train_params):] = config.rl.solver_sigma
  
  rl_logits = torch.nn.Parameter(init_tensor, requires_grad=True)
  train_params.append({"params": rl_logits, "lr": config.rl.lr})

  optimizer = optim.Adam(train_params)

  return rl_logits, optimizer
    


def get_grad_stats(grad, suffix):
  return {
    f'{suffix}/norm': grad.norm(2).item(),
    f'{suffix}/mean': grad.mean().item(),
    f'{suffix}/std':  grad.std().item()
  }

def grad_clip(config, rl_logits, timesteps_model, pipe):
  log_dict = {}

  if config.rl.train:
    log_dict.update(get_grad_stats(rl_logits.grad, 'grad_rl'))
    torch.nn.utils.clip_grad_norm_([rl_logits], max_norm=config.rl.clip_grad_norm)

  if config.timesteps.train:
    log_dict.update(get_grad_stats(timesteps_model.timesteps_logits.grad,      'grad_ts'))
    log_dict.update(get_grad_stats(timesteps_model.unet_timesteps_logits.grad, 'grad_uts'))
    torch.nn.utils.clip_grad_norm_([timesteps_model.timesteps_logits, timesteps_model.unet_timesteps_logits], max_norm=config.timesteps.clip_grad_norm)
      
  if config.solver.train:
    if isinstance(pipe.scheduler.train_params, torch.Tensor):
      log_dict.update(get_grad_stats(pipe.scheduler.train_params.grad, 'grad_solv'))
    else: # it is list of 1-d tensors
      solv_grad = torch.cat([i.grad for i in pipe.scheduler.train_params], dim=0)
      log_dict.update(get_grad_stats(solv_grad, 'grad_solv'))

    torch.nn.utils.clip_grad_norm_(list(pipe.scheduler.train_params), max_norm=config.solver.clip_grad_norm)
  
  return log_dict
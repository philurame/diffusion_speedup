from lib.registries import scheduler_registry
from lib.schedulers.mixin_scheduler import SchedulerMixin

@scheduler_registry.add_to_registry("STUDENTDEIS40")
class STUDENTDEIS40Scheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
    if num_inference_steps == 5:
      timesteps = [999.0000, 917.7686, 829.6602, 674.2077, 404.3070]
    elif num_inference_steps == 7:
      timesteps = [999.0000, 959.2513, 910.1037, 844.2115, 751.3869, 586.9683, 291.7593]
    elif num_inference_steps == 8:
      timesteps = [999.0, 937.361083984375, 870.4226684570312, 794.496826171875, 692.6321411132812, 510.07373046875, 311.46063232421875, 137.58123779296875]
    elif num_inference_steps == 9:
      timesteps = [999.0, 955.4049682617188, 903.2561645507812, 853.7620239257812, 793.8970947265625, 690.4998168945312, 516.7586669921875, 291.1212158203125, 110.97943115234375]
    elif num_inference_steps == 10:
      timesteps = [999.0000, 958.5742, 920.4229, 883.0010, 843.2300, 773.5538, 677.8006, 527.4633, 281.4029,  86.1095]
    elif num_inference_steps == 14:
      timesteps = [999.0000, 961.9507, 924.4011, 884.3823, 848.0552, 805.2089, 754.8856, 677.1332, 562.4333, 433.4813, 317.2736, 213.7070, 120.4547,  38.4559]
    else:
      raise NotImplementedError

    self.prepare_solver_data(timesteps, device)


@scheduler_registry.add_to_registry("STUDENTDDIM200")
class STUDENTDDIM200Scheduler(SchedulerMixin):
  def set_timesteps(self, num_inference_steps=None, device=None, **kwargs):    
    if num_inference_steps == 5:
      timesteps = [999.0000, 916.5345, 834.0232, 680.7986, 407.1682]
    elif num_inference_steps == 7:
      timesteps = [999.0000, 964.5082, 928.7987, 876.7093, 801.4840, 667.8565, 395.6930]
    elif num_inference_steps == 10:
      timesteps = [999.0000, 971.8139, 939.7130, 903.7587, 857.7821, 802.9174, 718.3980, 594.1359, 366.2291,  87.9242]
    elif num_inference_steps == 14:
      timesteps = [999.0000, 978.0498, 949.2379, 933.2876, 913.7000, 891.7931, 866.8490, 829.7654, 783.8227, 707.5926, 584.6900, 382.9999, 209.9362,  66.6091]
    else:
      raise NotImplementedError

    self.prepare_solver_data(timesteps, device)
import wandb, os
from PIL import Image
import pandas as pd

def is_already_calculated(save_path, calculated_path):
  cacher_quantizer, dataset_nfe, solver_scheduler = save_path.split('/')[-3:]
  dataset, nfe = dataset_nfe.split('_')
  solver, scheduler = solver_scheduler.replace('.pt', '').split('_')
  if not os.path.exists(calculated_path):
    return True
  df_already_calculated = pd.read_csv(calculated_path, index_col=0)
  res = df_already_calculated.query(
    "dataset == @dataset and \
    solver == @solver and \
    scheduler == @scheduler and \
    cacher_quantizer == @cacher_quantizer and \
    nfe == @nfe"
    ).shape[0] == 0
  return not res

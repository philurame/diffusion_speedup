# PROJECT STRUCTURE
- `pip install -r requirements.txt`
- Launch via `python run.py --generate X --wandb_key Y` based on config [config_generate.yaml](config_generate.yaml) (or [config_metric.yaml](config_metric.yaml)), `generate` boolean X and `wandb_key` str Y
- Usage examples in `notebooks`

## `COCO2017 & PARTI dataset`
- can be downloaded in [this kaggle dataset](https://www.kaggle.com/datasets/philurame/coco2017-and-parti)
- after downloading put it in the `DATA/datasets_coco_parti.pkl` path

## `EDM` (and CIFAR metrics)
- download [inceptionV3](https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl) and put it the `DATA/InceptionV3.pkl` path
- download [EDM model](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl) and put it the `DATA/edm-cifar10-32x32-uncond-vp.pkl` path

## [run.py](run.py), [config_generate.yaml](config_generate.yaml) -> [main_generate.py](main_generate.py)
### [run.py](run.py) 
- takes product of all compbinations from [config_generate.yaml](config_generate.yaml) (or [config_metric.yaml](config_metric.yaml)) and executes [main_generate.py](main_generate.py) (or [main_metric.py](main_metric.py)) with sbatch
- requires `wandb_key` and `generate` parameters

### [config_generate.yaml](config_generate.yaml) ([config_metric.yaml](config_metric.yaml))
- Configuration file for solvers, schedulers, cachers/quantizers

### [main_generate.py](main_generate.py)
- generates latent images for `SDXL` and stores in [DATA/...](DATA)

### [main_metric.py](main_metric.py)
- calculates metrics for stored latents and logs them into `wandb`

## [solvers](lib/solvers)
- Contains all solvers implementations
- Solvers include additional attributes for interaction with different schedulers

## [schedulers](lib/schedulers)
- Contains all schedulers implementations
- Schedulers must override `set_timesteps` based on their algorithm

## [models](lib/models)
- Contains all models (including EDM, SDXL) and their acceleration methods
- `models` methods (like cachers and quantizers) usually modify `unet` and override `_call_impl` method (alias of `__call__`)


# Implementation Details

## Datasets (`PARTI`, `COCO`)
- Downloaded via [kaggle](https://www.kaggle.com/code/philurame/downloading-cifar10-imagenet-mscoco-datasets)
- Postprocessed with [noteboook](notebooks/preprocess_datasets.ipynb) (to be added to Kaggle later): trims prompts (merges in the case of COCO) to fit within 77 tokens for the SDXL CLIP encoder

## Solvers

### `DDIM`

![alt text](_tex_imgs/DDIM.png)
<!-- $$x_{t-1}=\sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t+\sqrt{\alpha_{t-1}}\ \hat\varepsilon_{t}(\sigma_{t-1}-\sigma_t),\quad \sigma_t=\sqrt{1-\alpha_t}$$ -->

### `DPMS`: DpmSolver++(2M)
![alt text](_tex_imgs/DPMS.png)
<!-- $$x_{t-1}={\dfrac{\sigma_{t-1}}{\sigma_t}}x_t-\sqrt{\alpha_{t-1}}(e^{\lambda_t-\lambda_{t-1}}-1)D_t$$
$$D_t=(1+k_t)\hat x_t-k_t\hat x_{t+1},\quad \hat x_{t}=\frac{1}{\sqrt{\alpha_{t}}}(x_t-\sigma_t\hat\varepsilon_{t}),\quad k_t=\frac{\lambda_t-\lambda_{t-1}}{2(\lambda_{t+1}-\lambda_{t})},\quad \lambda_t=\log\frac{\sqrt{\alpha_t}}{\sigma_t}$$ -->

### `DEIS`: Exponential Integrator, "2 order" (which is actually 1th)
![alt text](_tex_imgs/DEIS.png)
<!-- $$x_{t_{i-1}}=\sqrt{\dfrac{\alpha_{t_{i-1}}}{\alpha_{t_{i}}}}x_{t_i}+C_0\hat\varepsilon_{t_i}+C_1\hat\varepsilon_{t_{i+1}}$$
$$C_k=\frac12\int_{t_{i}}^{t_{i-1}}\sqrt{\dfrac{\alpha_{t_{i-1}}}{\alpha_{\tau}}}(-\dfrac{d\log\alpha_{\tau}}{d\tau})\dfrac{1}{\sigma_\tau}P_k(\tau) d\tau,\ P_1(\tau)=\dfrac{\tau-t_{i+1}}{t_i-t_{i+1}},\ P_2(\tau)=\dfrac{\tau-t_{i}}{t_{i+1}-t_{i}}$$ -->

### `UNIPC`: Unified Predictor-Corrector, p=2
- possible to use "Corrector" with other solvers
- github cannot render TeX below

![alt text](_tex_imgs/UNIPC.png)
<!-- $$\text{predictor:}\  \tilde{x}_{t_i}=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{x}_{t_{i-1}}+\alpha_{t_i}\left(1-e^{-h_i}\right) x_\theta\left(\tilde{x}_{t_{i-1}}, t_{i-1}\right)+\alpha_{t_i} (e^{h_i}-1) \sum_{m=1}^{p-1} a_m^p D_m^x$$
$$\text{corrector:}\ \tilde{x}_{t_i}^c=\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{x}_{t_{i-1}}+\alpha_{t_i}\left(1-e^{-h_i}\right) x_\theta\left(\tilde{x}_{t_{i-1}}, t_{i-1}\right)+\alpha_{t_i} (e^{h_i}-1) \sum_{m=1}^p a_m^{p-1} D_m^x$$
$$a_m^p=R^{-1}_p(h_i)g_p(h_i)/(e^{h_i}-1),\ D_m^x=x_\theta(t_i-m) - x_\theta(t_{i-1})$$ -->


## Schedulers
![alt text](_tex_imgs/SNR.png)
<!-- - SNR $\sigma_i:=\sqrt{\dfrac{1-\tilde\alpha_{i}}{\tilde\alpha_{i}}},\ \hat\alpha_i=\prod\limits_{k\leq i}\alpha_k,\ \alpha_0\approx 1$ -->

### `FREEU`
rebalancing the contributions from the UNet’s skip connections and backbone feature maps by:
- normalization of backbone params
- weakening of high-frequency features in skip-connection using fourier domain

### `LINEAR`
![alt text](_tex_imgs/LINEAR.png)
<!-- - $t_i=\lfloor999 - i\frac{999}{\text{NFE}}\rceil$ -->

### `DDIM`: original time scheduling for DDIM
![alt text](_tex_imgs/DDIM_S.png)
<!-- - $t_i=(\text{NFE}-1-i)\cdot\lfloor \frac{1000}{\text{NFE}}\rceil$ -->

### `AYS`: Align Your Steps
- `[999, 845, 730, 587, 443, 310, 193, 116, 53, 13]` for `NFE = 10`
- [loglinear interpolation](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html) timesteps for `NFE != 10`

### `KARRAS`: "Karras Sigmas" from EDM
![alt text](_tex_imgs/KARRAS.png)
<!-- - $\hat\sigma_i=\left(\sigma_{\max}^{1/p}+\frac{i}{\text{NFE}-1}(\sigma_{\min}^{1/p}- \sigma_{\max}^{1/p})\right)^p,\quad p=7$
- $t_i=\underset{k}{\text{argmin }}|\log\hat\sigma_i-\log\sigma_k|$ -->

### `SNR`: LogSNR uniform from DPMSolver
![alt text](_tex_imgs/SNR_S.png)
<!-- - $\hat\sigma_i=\log\sigma_{\max}+\frac{i}{\text{NFE}-1}(\log\sigma_{\min}-\log\sigma_{\max})$
- $t_i=\underset{k}{\text{argmin }}|\hat\sigma_i-\sigma_k|$ -->

## Cachers, Quantizers

### `DEEPCACHE`
- caching unet starting from `cache_branch_id` skip connection and reusing it every `cache_interval` iterations (making full inference only 1 of cache_interval consecutive iterations)
- hyperparameters should better be optimized

### `TGATE`: T-Gate (gate_step = NFE//2.5)
- caching cross attentions outcomes at the `gate_step` timestep and reusing them at at subsequent ("fidelity-improving phase") steps
- the choice of parameter `m` (or `gate_step`) is due to interpolation of the “optimal” parameters from the paper: `(NFE, m)` = (15, 6), (25, 10), (50, 20), (100, 25) => `m ~ NFE//2.5`.
- hyperparameters should better be optimized

### `HQQ`: Half-Quadratic Quantization
- quantization of weights w.o. calibration data

![alt text](_tex_imgs/HQQ.png)
<!-- $$\underset{W_e,\ Z,\ S}{\min}\left[\|W_e\|_{p<1}+\beta\|W_e-(W_f-\hat W_f(S,Z))\|^2_F\right]$$
$$\hat W_f(S,Z)=S(\hat W - Z),\quad \hat W=\lfloor W S^{-1} + Z\rceil,\ \hat W-\text{4bit}$$ -->


### `VQDM`: Vector Quantized Diffusion Model
- quantization of weights based on calibration data

![alt text](_tex_imgs/VQDM.png)
<!-- $$\underset{C, b}{\min } \| \mathbf{W X}-\left(\text { Concat }_{i, j} \sum_{m=1}^M C_m b_{i, j, m}\right) \mathbf{X} \|_2^2$$ -->
- gradient optimization of float codebooks $C_m$ and beam search for discrete $b_{i,j,m}$ index-vectors
- takes ~16hrs on A100 (no finetune)
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def sample_exp(logits, inference=False):
    """
    Сэмплирует значения по Exp-Min Trick.
    
    Для каждой i-ой компоненты получаем случайную величину G_i = g_i + l_i,
    где l_i - это i-й логит; g_i=log(-log(U_i)), U_i ~ Uniform(0, 1) - шум из стандартного распределения Гумбеля
    """
    if inference:
        return torch.exp(logits)

    uniform_noise = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    exp_noise = torch.log(-torch.log(uniform_noise))
    
    sampled_values = torch.exp(logits + exp_noise)
    sampled_values.requires_grad_(True)
    
    return sampled_values

def top_k_log_prob(logits, perturbed_logits, k):
    """
    Вычисляет log-prob для первых k элементов упорядоченной последовательности логитов.

    log P(t) = log P(t_1, ..., t_k) для засэмплированной последовательности t.
    
    P(t) = ∏_{i=1..k} [ exp(θ_{t_i}) / (Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(θ_j)) ]
    
    log P(t) = Σ_{i=1..k} [ exp(θ_{t_i}) - log(Σ_{j ∉ {t_1, ..., t_{i-1}}} exp(θ_j)) ], где θ - это наши `logits`.

    Args:
        logits - исходный вектор логитов
        perturbed_logits - зашумленные логиты из sample_gumbel_max
        k - количество логитов, для которых считаем log-prob

    Returns:
        tuple(
            Индексы первых k элементов в отсортированной последовательности,
            log P(t_1, ..., t_k)
        )
    """
    num_total_steps = logits.shape[0] # (N, )
    
    # Получаем упорядоченную последовательность t = (t_1, ..., t_k), что эквивалетно взятию argmin k раз.
    indices = torch.argsort(perturbed_logits)
    top_k_indices = indices[:k] # (k, )
    
    # Создаем бинарную маску, где bin_mask[i, j] = 1, если argmins[i] = j
    all_indices = torch.arange(num_total_steps, device=logits.device)
    bin_mask = (top_k_indices[:, None] == all_indices[None, :]) # (k, N)
    
    # Вычисляем числитель (выбираем нужный логит для каждого шага i)
    log_numerator = - (logits[None, :] * bin_mask).sum(dim=-1) # (k,)

    # Вычисляем знаменатель
    # Создаем маску, которая на шаге i будет выключать уже выбранные элементы t_1, ..., t_{i-1}.
    exclusion_mask = torch.cumsum(bin_mask, dim=0)[:-1, :] # (k-1, N)
    # Добавляем строчку нулей, которая говорит, что на шаге 1 мы ничего не исключаем
    exclusion_mask = torch.cat([torch.zeros(1, num_total_steps, device=logits.device), exclusion_mask], dim=0) # (k, N)    
    # Применяем маску: добавляем -inf к уже выбранным логитам, чтобы они не участвовали в logsumexp.
    exclusion_mask[exclusion_mask > 0] = float('inf')
    masked_logits = -logits[None, :] - exclusion_mask
    log_denominator = torch.logsumexp(masked_logits, dim=-1) # (k,) 
    
    # Собираем все вместе
    log_prob = (log_numerator - log_denominator).sum() 

    # Мы передаем в cacher шаги, которые пропускаем
    return indices[k:] + 1, log_prob
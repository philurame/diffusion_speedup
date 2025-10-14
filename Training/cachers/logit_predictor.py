import torch
import torch.nn as nn
from munch import Munch

class BaseLogitModel(nn.Module):
    def __init__(self, n_logits: int, config: Munch, dtype=torch.float32, device=torch.device('cuda')):
        super().__init__()
        
        self.dtype = dtype
        self.config = config
        
        self.logits = torch.nn.Parameter(
            self.init_logits(n_logits), 
            requires_grad=config.train_logits
        )
        self.to(device)
        
    def init_logits(self, n_logits):
        if self.config.init_logits == 'ones':
            return torch.ones(n_logits, dtype=self.dtype) 
        
        if self.config.init_logits == 'randn':
            return torch.randn(n_logits, dtype=self.dtype) 
        
        if self.config.init_logits == 'zeros':
            return torch.zeros(n_logits, dtype=self.dtype) 
        
        if self.config.init_logits == 'deepcache-3':
            logits = torch.ones(n_logits, dtype=self.dtype) 
            logits[2::3] *= 0.9 # пересчитываемые логиты должны быть поменьше
            return logits
        if self.config.init_logits == 'deepcache-6':
            logits = torch.ones(n_logits, dtype=self.dtype) 
            logits[0::6] *= 0.9 # пересчитываемые логиты должны быть поменьше
            return logits
        
        if self.config.init_logits == 'deepcache-3 smaller diff':
            logits = torch.ones(n_logits, dtype=self.dtype) 
            logits[2::3] *= 0.975
            return logits
        
        # if self.config.init_logits == 'deepcache-3 around 0':
        #     logits = torch.zeros(n_logits, dtype=self.dtype) 
        #     logits -= 0.05
        #     logits[2::3] += 0.1
        #     return logits
        
        if self.config.init_logits == 'deepcache-4':
            logits = torch.ones(n_logits, dtype=self.dtype) 
            logits[3::4] *= 0.9 # пересчитываемые логиты должны быть поменьше ???
            return logits
        
        if self.config.init_logits == 'mishans':
            logits = torch.ones(n_logits, dtype=self.dtype) 
            to_recalculate = [0, 2, 4, 7, 10, 15, 20, 23]
            logits[to_recalculate] *= 0.9 
            return logits
        
        raise ValueError(f"Unknown init_logits type: {self.config.init_logits}")
    
    def model_parameters(self):
        return list()
        
    def forward(self, _):
        return self.logits

class SmallMLP(BaseLogitModel):
    def __init__(self, n_logits: int, config: Munch, dtype=torch.float32, device=torch.device('cuda')):
        super().__init__(n_logits, config, dtype, device)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, n_logits)
        )
        self.mlp_head.to(device)
        
        for parameter in self.mlp_head.parameters():
            nn.init.zeros_(parameter)
    
    def extract_prompt_embeddings(self, prompt_embeddings):
        # ( 
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds,
        # ) 
        
        if self.config.prompt_extraction_variant == 'pooled_prompt_embeds':
            return prompt_embeddings[2]
        
        NotImplementedError(f'unknown prompt_extraction_variant passed, {self.config.prompt_extraction_variant}')
        
    def model_parameters(self):
        return self.mlp_head.parameters()
    
    def forward(self, prompt_embeddings):
        prompt_embeddings = self.extract_prompt_embeddings(prompt_embeddings)
        prompt_embeddings = prompt_embeddings.to(dtype=self.dtype)

        predicted = self.mlp_head(prompt_embeddings).squeeze(0)
        
        return self.logits + predicted
    
    

def load_logit_model(n_logits, config, dtype, device) -> BaseLogitModel:
    if config.model_variant == 'constant':
        return BaseLogitModel(n_logits, config, dtype, device)
        
    if config.model_variant == 'SmallMLP':
        return SmallMLP(n_logits, config, dtype, device)
    
    raise NotImplementedError(f'unknown logit model passed, {config.model_variant}')











# # хочу cls token pooling
# # еще не хочу делать positional embeddings 
# self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# encoder_layer = nn.TransformerEncoderLayer(
#     d_model=embed_dim, 
#     nhead=config.nhead_encoder, 
#     dim_feedforward=config.dim_feedforward_encoder, 
#     dropout=config.dropout,
#     batch_first=True
# )
# self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.nlayer_encoder)

# self.mlp_head = nn.Sequential(
#     nn.Linear(embed_dim, 512),
#     nn.ReLU(),
#     nn.Dropout(config.dropout),
#     nn.Linear(512, n_logits)
# )

# # если мы хотим обрабатывать еще разное количество шагов, 
# # то можно просто сделать декодер и кайфовать
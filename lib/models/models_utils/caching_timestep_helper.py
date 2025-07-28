from contextlib import contextmanager
from typing import List

class CachingTimestepHelper(object):
    def __init__(self, pipe):
        self.pipe = pipe    
            
        self.function_dict = dict()   
        self.cached_output = dict()
        self.schedule = list()
        self.timestep = 0
        
    @contextmanager
    def default(self):
        self.timesteps = [] # timestep, которые мы не будем пересчитывать
        yield
  
    @contextmanager
    def inference(self, timesteps: List):
        self.timesteps = timesteps # timestepы, которые мы не будем пересчитывать
        if not self.function_dict:
            self.wrap_modules()

        self.schedule = list()
        self.recalculate = True       
        yield
        self.cached_output = dict()
        
    def wrap_unet(self):
        self.function_dict['unet_forward'] = self.pipe.unet.forward
        
        def wrapped_forward(*args, **kwargs):
            self.timestep = list(self.pipe.scheduler.timesteps).index(args[1].item())
            
            # print(f'\n{self.timestep}\n')
            self.recalculate = self.timestep not in self.timesteps
            self.schedule.append(self.recalculate)
            
            output = self.function_dict['unet_forward'](*args, **kwargs)
            return output
        
        self.pipe.unet.forward = wrapped_forward

    def wrap_blocks(self, block, block_name, block_i, layer_i, blocktype = "down"):
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward
        
        def wrapped_forward(*args, **kwargs):            
            if (blocktype, block_name, block_i, layer_i) in [('up', 'resnet', 0, 0), ('up', 'block', 0, 0)]:
                return self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
            
            if self.recalculate:
                self.cached_output[(blocktype, block_name, block_i, layer_i)] = self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
                
            return self.cached_output[(blocktype, block_name, block_i, layer_i)]
        
        block.forward = wrapped_forward
        
    def wrap_modules(self):
        # 1. wrap unet
        self.wrap_unet()
        
        # 2. wrap downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_blocks(attention, "attentions", block_i, layer_i)
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_blocks(resnet, "resnet", block_i, layer_i)
            for downsampler in getattr(block, "downsamplers", []) if block.downsamplers else []:
                self.wrap_blocks(downsampler, "downsampler", block_i, len(getattr(block, "resnets", [])))
            self.wrap_blocks(block, "block", block_i, 0, blocktype = "down")
        
        # 3. wrap midblock forward
        self.wrap_blocks(self.pipe.unet.mid_block, "mid_block", 0, 0, blocktype = "mid")
        
        # 4. wrap upblock forward
        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_blocks(attention, "attentions", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_blocks(resnet, "resnet", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for upsampler in getattr(block, "upsamplers", []) if block.upsamplers else []:
                self.wrap_blocks(upsampler, "upsampler", block_num - block_i - 1, 0, blocktype = "up")
            self.wrap_blocks(block, "block", block_num - block_i - 1, 0, blocktype = "up")
            
    def unwrap_modules(self):
        # 1. unet forward
        self.pipe.unet.forward = self.function_dict['unet_forward']
        
        # 2. downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("down", "attentions", block_i, layer_i)]

        # 3. midblock forward
        for (layer_i, attention) in enumerate(self.pipe.unet.mid_block.attentions):
            attention.forward = self.function_dict[("mid", "attentions", 0, layer_i)]
            
        # 4. upblock forward
        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("up", "attentions", block_num - block_i - 1, layer_num - layer_i - 1)]
            
        self.function_dict = dict()
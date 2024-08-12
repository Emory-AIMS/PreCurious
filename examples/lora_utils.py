import torch
import torch.nn as nn
import transformers



class lora_linear(nn.Module):


    def __init__(self, layer, r, alpha):
        super(lora_linear, self).__init__()
        self.main_branch = layer
        self.lora_scaling = alpha / r

        if isinstance(layer, transformers.adapters.lora.MergedLinear):
            self.lora_branch_in = nn.Linear(layer.weight.shape[0], r, bias=False)
            self.lora_branch_out = nn.Linear(r, layer.weight.shape[1], bias=False)
        else:
            self.lora_branch_in = nn.Linear(layer.weight.shape[1], r, bias=False)
            self.lora_branch_out = nn.Linear(r, layer.weight.shape[0], bias=False)
        
        # zero init lora_branch_out
        self.lora_branch_out.weight.data.zero_()

        # move lora_branch_in to the same device as main_branch
        self.lora_branch_in.to(layer.weight.device)
        self.lora_branch_out.to(layer.weight.device)

    def forward(self, x):
        # (Pdb) self.main_branch(x).shape
        # torch.Size([4, 1024, 2304])
        # (Pdb) self.main_branch.weight.shape
        # torch.Size([768, 2304])
        # (Pdb) x.shape                      
        # torch.Size([4, 1024, 768])
        # (Pdb) self.lora_branch_in.weight.shape
        # torch.Size([16, 2304])
        # (Pdb) self.lora_branch_out.weight.shape
        # torch.Size([768, 16])
        main_branch_out = self.main_branch(x)
        try:
            lora_branch_in = self.lora_branch_in(x) # torch.Size([4, 1024, 16])
            lora_branch_out = self.lora_branch_out(lora_branch_in) * self.lora_scaling
        except:
            import pdb; pdb.set_trace()
        return main_branch_out + lora_branch_out
    

def make_lora(model, model_name, r=16, alpha=16, lora_weights=['q', 'k', 'v', 'o']):
    if 'llama' in model_name:
        make_lora_llama(model, r, alpha, lora_weights)
    elif 'gpt2' in model_name:
        make_lora_gpt2(model, r, alpha, lora_weights)
    else:
        raise NotImplementedError

def make_lora_gpt2(model, r, alpha=16, lora_weights = ['c_attn', 'c_proj']):
    # make attention weights lora
    for layer in model.transformer.h:
        for w in lora_weights:
            if w == 'c_attn':
                layer.attn.c_attn = lora_linear(layer.attn.c_attn, r, alpha)
            elif w == 'k':
                layer.attn.c_proj = lora_linear(layer.attn.c_proj, r, alpha)

    # make the lm_head lora
    if isinstance(model.lm_head, nn.Linear):
        model.lm_head = lora_linear(model.lm_head, r, alpha)
    else:
        raise 'lm_head is not linear, need to check its implementation and adapt the codebase'

    num_trainable_p = 0

    for i, p in enumerate(model.named_parameters()):
        if 'lora_branch' in p[0]:
            p[1].requires_grad = True
            num_trainable_p += p[1].numel()
        else:
            p[1].requires_grad = False

def make_lora_llama(model, r, alpha, lora_weights):

    layers = model.layers if hasattr(model, 'layers') else model.model.layers

    # make attention weights lora
    for layer in layers:
        for w in lora_weights:
            if w == 'q':
                layer.self_attn.q_proj = lora_linear(layer.self_attn.q_proj, r, alpha)
            elif w == 'k':
                layer.self_attn.k_proj = lora_linear(layer.self_attn.k_proj, r, alpha)
            elif w == 'v':
                layer.self_attn.v_proj = lora_linear(layer.self_attn.v_proj, r, alpha)
            elif w == 'o':
                layer.self_attn.o_proj = lora_linear(layer.self_attn.o_proj, r, alpha)

    # make the lm_head lora
    # if isinstance(model.lm_head, nn.Linear):
    #     model.lm_head = lora_linear(model.lm_head, r, alpha)
    # else:
    #     raise 'lm_head is not linear, need to check its implementation and adapt the codebase'

    num_trainable_p = 0

    for i, p in enumerate(model.named_parameters()):
        if 'lora_branch' in p[0]:
            p[1].requires_grad = True
            num_trainable_p += p[1].numel()
        else:
            p[1].requires_grad = False
            
def load_weights(lora_model, lora_weights_path):
    lora_weights = torch.load(lora_weights_path)
    lora_model.load_state_dict(lora_weights, strict=False)

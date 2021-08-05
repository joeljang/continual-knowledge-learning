from transformers import *
import torch
import numpy as np
import torch.nn.utils.prune as prune

def count_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    
class ModelPruner():
    def __init__(self, base_pretrained_model_name=‘t5-small’, prune_percent=0.2):
        self.name = base_pretrained_model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.name)
        import pdb; pdb.set_trace()
        self.prune_percent = prune_percent
    def count_nonzero_param(self):
        nz = 0
        for param in self.model.parameters():
            nz += torch.nonzero(param).size(0)
        return nz
    def prune_and_save(self):
        p = prune.L1Unstructured(amount=self.prune_percent)
        orig_param = self.count_nonzero_param()
        for name, param in self.model.named_parameters():
            param.data = p.prune(param.data)
        aft_param = self.count_nonzero_param()
        print(f”# nonzero parameters: {orig_param} -> {aft_param}“)
        torch.save(self.model.state_dict(), f”{self.name}-{self.prune_percent}.pt”)
        print(“Prune and save done!“)
        print(f’Instructions for loading this model:\n>>>model = T5ForConditionalGeneration.from_pretrained(“{self.name}“)\n\

model.load_state_dict(torch.load(“{self.name}-{self.prune_percent}.pt”))\n\
model.eval() # optional’)

def wip():
    model = T5Model.from_pretrained(‘t5-small’)
    #param_list = ()
    count_param(model)
    count_nonzero_param(model)
    #output = prune.global_unstructured(param_list, pruning_method=prune.L1Unstructured, amount=0.2)
    p = prune.L1Unstructured(amount=0.2)
    #pruned = p.prune(model)
    #sp = pruned.to_sparse()
    for name, param in model.named_parameters():
        print(param)
        param.data = p.prune(param.data)
        print(param)
        print(‘\n\n\n\n\n\n’)
    #print(model.decoder.final_layer_norm.weight)
    count_param(model)
    count_nonzero_param(model)
    import pdb; pdb.set_trace()
    
if __name__ == ‘__main__‘:
    # t5-base (->small): 1-0.27144755414
    # t5-large (->base): 1-0.30217323103
    pruner = ModelPruner(base_pretrained_model_name=‘t5-base’, prune_percent= 1-0.27144755414)
    pruner.prune_and_save()
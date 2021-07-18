from models.t5_baseline import T5 as T5_Baseline
from models.t5_modular import T5 as T5_Modular
from models.t5_modular_small import T5 as T5_Modular_Small
from models.t5_adapters import T5 as T5_adapter
from models.t5_recadam import T5 as T5_Recadam
from models.t5_mixreview import T5 as T5_MixReview
from models.t5_modular_layerwise import T5 as T5_Modular_LW
from models.t5_biasonly import T5 as T5_BiasOnly
from models.t5_lora import T5 as T5_Lora

from models.gpt2_baseline import GPT2 as GPT2_Baseline

T5_MODELS = {
    'models_baseline': T5_Baseline,
    'models_modular': T5_Modular,
    'models_modular_small': T5_Modular_Small,
    'models_adapter': T5_adapter,
    'models_recadam': T5_Recadam,
    'models_mixreview': T5_MixReview,
    'models_modular_lw' : T5_Modular_LW,
    'models_biasonly' : T5_BiasOnly,
    'models_lora' : T5_Lora
}

GPT2_MODELS = {
    'models_baseline': GPT2_Baseline,
    #'models_adapter': GPT2_adapter,
    #'models_recadam': GPT2_Recadam,
    #'models_mixreview': GPT2_MixReview,
}

def load_model(name: str, type: str):
    if type=='T5':
        return T5_MODELS[name]
    elif type=='GPT2':
        return GPT2_MODELS[name]
    else:
        raise Excpetion('Select the correct model type. Currently supporting only T5 and GPT2.')
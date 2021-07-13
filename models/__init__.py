from models.models_baseline import T5FineTuner as Models_Baseline
from models.models_modular import T5FineTuner as Models_Modular
from models.models_modular_small import T5FineTuner as Models_Modular_Small
from models.models_adapters import T5FineTuner as Models_adapter
from models.models_recadam import T5FineTuner as Models_Recadam
from models.models_mixreview import T5FineTuner as Models_MixReview

MODELS = {
    'models_baseline': Models_Baseline,
    'models_modular': Models_Modular,
    'models_modular_small': Models_Modular_Small,
    'models_adapter': Models_adapter,
    'models_recadam': Models_Recadam,
    'models_mixreview': Models_MixReview
}

def load_model(name: str):
    return MODELS[name]
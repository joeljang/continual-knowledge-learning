#import models_baseline, models_modular, models_modular_small, models_kadapter, models_recadam
from models.models_baseline import T5FineTuner as Models_Baseline
from models.models_modular import T5FineTuner as Models_Modular
from models.models_modular_small import T5FineTuner as Models_Modular_Small
from models.models_kadapter import T5FineTuner as Models_Kadapter
from models.models_recadam import T5FineTuner as Models_Recadam

MODELS = {
    'models_baseline': Models_Baseline,
    'models_modular': Models_Modular,
    'models_modular_small': Models_Modular_Small,
    'models_kadapter': Models_Kadapter,
    'models_recadam': Models_Recadam,
}

def load_model(name: str):
    return MODELS[name]
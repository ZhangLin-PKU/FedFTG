from models import resnet
from models import simple_models
from models import vgg
from models import gan_models


def choose_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnet18(**kwargs)
    else:
        raise ValueError('Wrong model name.')

def choose_g_model(model_name, **kwargs):
    if model_name == 'GeneratorA':
        return gan_models.GeneratorA(**kwargs)
    else:
        raise ValueError('Wrong model name.')

from .MSV863_text_cot import MSV863_Text
from .RGBNT201_Text_cot import RGBNT201_Text
from .msvr310_Text import MSVR310_text


__factory = {
    'RGBNT201': RGBNT201_Text,
    'MSVR310': MSVR310_text,
    'WMVEID863': MSV863_Text,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

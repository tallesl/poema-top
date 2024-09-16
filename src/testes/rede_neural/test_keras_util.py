# ignorando o seguinte warning do TensorFlow:
# DeprecationWarning: The distutils package is deprecated and slated for removal in Python 3.12. Use setuptools or check
# PEP 632 for potential alternatives
from warnings import filterwarnings
filterwarnings('ignore', message=".*distutils package is deprecated.*", category=DeprecationWarning)

from poema_top.rede_neural.keras_util import obtem_vram

def teste_obtem_vram():
    # act
    vram_inicial, vram_pico_inicial = obtem_vram()

    # assert
    assert vram_inicial is not None
    assert vram_pico_inicial is not None

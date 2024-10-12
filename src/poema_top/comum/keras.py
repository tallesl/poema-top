from os import listdir, path
from typing import Callable, Optional

from keras.callbacks import Callback
from keras.models import load_model, Model
import tensorflow as tf

from . import configuracao


def alocar_memoria_aos_poucos() -> None:
    '''
    Evita com que o TensorFlow aloque toda a memória da GPU ao iniciar.
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


def carrega_ultimo_modelo() -> Model:

    # listando todos arquivos .keras do diretório
    diretorio = configuracao.diretorio_checkpoint
    arquivos = listdir(diretorio)
    arquivos_keras = [path.join(diretorio, a) for a in arquivos if a.endswith('.keras')]

    # se nenhum arquivo .keras foi encontrado
    if not arquivos_keras:
        raise FileNotFoundError(f'Nenhum arquivo .keras encontrado no diretório: {diretorio}')

    # obtém o arquivo com a data de alteração mais recente
    ultimo_modelo = max(arquivos_keras, key=path.getmtime)

    # carrega o modelo
    modelo = load_model(ultimo_modelo)

    print(f'Carregado modelo "{ultimo_modelo}".')

    return modelo


def obtem_vram() -> tuple[int, int]:
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu = gpus[0] # utilizando apenas a primeira GPU

    device = gpu.name.lstrip('/physical_device:')
    memory_info = tf.config.experimental.get_memory_info(device)

    vram_atual_mb = memory_info['current'] / 1024**2
    vram_pico_mb = memory_info['peak'] / 1024**2

    return vram_atual_mb, vram_pico_mb

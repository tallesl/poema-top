'''
Módulo com funções que auxiliam no uso do Keras.
'''

from os import listdir
from os.path import getmtime, join

from keras.models import load_model, Model
import tensorflow as tf

from . import configuracao # pylint: disable=no-name-in-module

def alocar_memoria_aos_poucos() -> None:
    '''
    Evita com que o TensorFlow aloque toda a memória da GPU ao iniciar.
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def carrega_ultimo_modelo() -> Model:
    '''
    Obtém o último modelo da pasta de checkpoint, de acordo com a data de modificação do arquivo, carrega-o e retorna-o.
    '''

    # listando todos arquivos .keras do diretório
    diretorio = configuracao.diretorio_checkpoint
    arquivos = listdir(diretorio)
    arquivos_keras = [join(diretorio, a) for a in arquivos if a.endswith('.keras')]

    # se nenhum arquivo .keras foi encontrado
    if not arquivos_keras:
        raise FileNotFoundError(f'Nenhum arquivo .keras encontrado no diretório: {diretorio}')

    # obtém o arquivo com a data de alteração mais recente
    ultimo_modelo = max(arquivos_keras, key=getmtime)

    # carrega o modelo
    modelo = load_model(ultimo_modelo)

    print(f'Carregado modelo "{ultimo_modelo}".')

    return modelo

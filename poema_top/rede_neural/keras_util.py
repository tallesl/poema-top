from typing import Callable, Optional

import tensorflow as tf
from keras.callbacks import Callback


def alocar_memoria_aos_poucos():
    '''
    Evita com que o TensorFlow aloque toda a memória da GPU ao iniciar.
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

class CallbackFimEpoca(Callback):
    def __init__(self, callback: Callable[[], None], pular_epocas: Optional[int] = None):
        self.callback = callback
        self.pular_epocas = pular_epocas


    def on_epoch_end(self, epoch: int, logs: Optional[str] = None) -> None:
        if self.pular_epocas and (epoch % self.pular_epocas != 0):
            return

        self.callback()

import tensorflow as tf


def alocar_memoria_aos_poucos():
    '''
    Evita com que o TensorFlow aloque toda a memória da GPU ao iniciar.
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

from keras.models import load_model

from .pre_processamento.leitura_dataset import le_txt_dataset, extrai_janela
from .pre_processamento.vocabulario import Vocabulario

if __name__ == '__main__':
    texto_completo = le_txt_dataset()
    vocabulario = Vocabulario(texto_completo)
    modelo = load_model('../modelos_treinados/epoch-239-loss-0.8769665956497192.keras')

    from .rede_neural.redes import _gera_amostra
    _gera_amostra(modelo, vocabulario, texto_completo)

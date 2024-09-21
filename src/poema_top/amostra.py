from .log.log_util import LogaMemoria
from .pre_processamento.leitura_dataset import le_txt_dataset
from .pre_processamento.vocabulario import Vocabulario
from .rede_neural.keras_util import alocar_memoria_aos_poucos
from .rede_neural.redes import carrega_ultimo_modelo, gera_amostras

if __name__ == '__main__':
    alocar_memoria_aos_poucos()

    with LogaMemoria('Lendo txt...'):
        texto_completo = le_txt_dataset()

    with LogaMemoria('Montando vocabulário...'):
        vocabulario = Vocabulario(texto_completo)

    with LogaMemoria('Carregando último modelo...'):
        modelo = carrega_ultimo_modelo()

    with LogaMemoria('Gerando amostras...'):
        gera_amostras(modelo, vocabulario, texto_completo)

from . import configuracao
from .log.log_util import LogaMemoria
from .pre_processamento.janelas_one_hot import JanelasOneHot
from .pre_processamento.leitura_dataset import le_txt_dataset
from .pre_processamento.vocabulario import Vocabulario
from .rede_neural.keras_util import alocar_memoria_aos_poucos
from .rede_neural.redes import callbacks_treino, modelo_fchollet

if __name__ == '__main__':
    alocar_memoria_aos_poucos()

    with LogaMemoria('Lendo txt...'):
        texto_completo = le_txt_dataset()

    with LogaMemoria('Quebrando em janelas e codificando em one hot...'):
        vocabulario = Vocabulario(texto_completo)
        janelas = JanelasOneHot(texto_completo, vocabulario)

    with LogaMemoria('Instanciando modelo...'):
        modelo = modelo_fchollet(vocabulario)
        callbacks = callbacks_treino(modelo, vocabulario, texto_completo)

    with LogaMemoria('Treinando...'):
        modelo.fit(janelas.x, janelas.y, batch_size=128, epochs=configuracao.epocas_treino, callbacks=callbacks)

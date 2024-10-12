from datetime import datetime
from os.path import join

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GRU, Input
from keras.models import Model, Sequential # type: ignore[import-untyped]
from keras.optimizers import RMSprop # type: ignore[import-untyped]

from . import configuracao
from ..comum import configuracao as configuracao_comum
from ..comum.dataset import le_txt_dataset
from ..comum.keras import alocar_memoria_aos_poucos
from ..comum.log import LogaMemoria
from ..comum.vocabulario import Vocabulario
from .grafico_loss import GraficoLoss
from .janelas_one_hot import JanelasOneHot


def cria_modelo(vocabulario: Vocabulario) -> Model:
    # "a plain stack of layers where each layer has exactly one input tensor and one output tensor"
    modelo = Sequential()

    # camada de entrada
    formato_entrada = (configuracao_comum.tamanho_janela, vocabulario.tamanho)
    modelo.add(Input(shape=formato_entrada))

    # primeira camada GRU (oculta)
    modelo.add(GRU(128, return_sequences=True))

    # segunda camada GRU (oculta)
    modelo.add(GRU(128))

    # camada de saída
    modelo.add(Dense(vocabulario.tamanho, activation='softmax'))

    # atualiza os pesos a cada batch utilizando a média móvel dos quadrados dos gradientes das últimas N iterações,
    # ajustando as taxas de aprendizado (taxa individual para cada peso), prevenindo a explosão e dissipação dos
    # gradientes
    otimizador = RMSprop(learning_rate=configuracao.taxa_aprendizagem)
    modelo.compile(loss='categorical_crossentropy', optimizer=otimizador)

    return modelo


def callbacks_treino(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> list[Callback]:
    agora = datetime.now().strftime('%Y%m%d-%H%M%S')

    caminho_checkpoint = join(configuracao.diretorio_modelo, f'{agora}-epoch-{{epoch}}-loss-{{loss}}.keras')
    caminho_grafico = join(configuracao.diretorio_modelo, f'{agora}-loss.png')

    callback_checkpoint = ModelCheckpoint(filepath=caminho_checkpoint, monitor='loss', save_best_only=True)
    callback_grafico_loss = GraficoLoss(caminho_grafico)
    callback_parada = EarlyStopping(monitor='loss', patience=configuracao.paciencia_treino, verbose=1)

    return [callback_checkpoint, callback_grafico_loss, callback_parada]


def main() -> None:
    alocar_memoria_aos_poucos()

    with LogaMemoria('Lendo txt...'):
        texto_completo = le_txt_dataset()

    with LogaMemoria('Montando vocabulário...'):
        vocabulario = Vocabulario(texto_completo)

    with LogaMemoria('Quebrando em janelas e codificando em one hot...'):
        janelas = JanelasOneHot(texto_completo, vocabulario)

    with LogaMemoria('Instanciando modelo...'):
        modelo = cria_modelo(vocabulario)
        callbacks = callbacks_treino(modelo, vocabulario, texto_completo)

    with LogaMemoria('Treinando...'):
        modelo.fit(janelas.x, janelas.y, batch_size=128, epochs=configuracao.epocas_treino, callbacks=callbacks)


if __name__ == '__main__':
    main()

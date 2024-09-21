from random import randint
from sys import stdout

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
import numpy as np

from .. import configuracao
from ..pre_processamento.vocabulario import Vocabulario
from .keras_util import CallbackFimEpoca
from .predicao import seleciona_caractere

from .. import configuracao

def modelo_fchollet(vocabulario: Vocabulario) -> Model:
    # "a plain stack of layers where each layer has exactly one input tensor and one output tensor"
    modelo = Sequential(camadas)

    # camada de entrada
    formato_entrada = (configuracao.tamanho_janela, vocabulario.tamanho)
    modelo.layers.append(Input(shape=formato_entrada))

    # camada LSTM opcional
    if configuracao.duas_camadas_lstm:
        modelo.layers.append(LSTM(128, return_sequences=True))

    # camada LSTM sempre present
    modelo.layers.append(LSTM(128))

    # camada de saída
    modelo.layers.append(Dense(vocabulario.tamanho, activation='softmax'))

    # atualiza os pesos a cada batch utilizando a média móvel dos quadrados dos gradientes das últimas N iterações,
    # ajustando as taxas de aprendizado (taxa individual para cada peso), prevenindo a explosão e dissipação dos
    # gradientes
    otimizador = RMSprop(learning_rate=configuracao.taxa_aprendizagem)
    modelo.compile(loss='categorical_crossentropy', optimizer=otimizador)

    return modelo


def callbacks_treino(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> list[Callback]:
    callback_checkpoint = ModelCheckpoint(filepath=configuracao.caminho_checkpoint, monitor='loss', save_best_only=True)
    callback_amostra = CallbackFimEpoca(lambda: _gera_amostra(modelo, vocabulario, texto_completo), configuracao.epocas_amostra)
    callback_parada = EarlyStopping(monitor='loss', patience=configuracao.paciencia_treino, verbose=1)

    return [callback_checkpoint, callback_amostra, callback_parada]


def _gera_amostra(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> None:
    indice_comeco_ultima_janela = len(texto_completo) - configuracao.tamanho_janela - 1
    inicio_janela = randint(0, indice_comeco_ultima_janela)
    fim_janela = inicio_janela + configuracao.tamanho_janela

    janela_aleatoria = texto_completo[inicio_janela : fim_janela]

    for temperatura in [0.2, 0.6, 1.0]:
        print()
        print(f'Amostra com temperatura {temperatura}:')
        print(f'[{janela_aleatoria}]', end='', flush=True)

        janela_atual = janela_aleatoria

        for i in range(200):
            sampled = np.zeros((1, configuracao.tamanho_janela, vocabulario.tamanho))
            for i, char in enumerate(janela_atual):
                sampled[0, i, vocabulario.obtem_indice[char]] = 1.

            previsto = modelo.predict(sampled, verbose=0)[0]

            proximo_indice = seleciona_caractere(previsto, temperatura)
            proximo_caractere = vocabulario.obtem_caractere[proximo_indice]

            # remove primeiro caractere
            janela_atual = janela_atual[1:]

            # adiciona novo caractere predito
            janela_atual += proximo_caractere

            stdout.write(proximo_caractere)
            stdout.flush()

        print()

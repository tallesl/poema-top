from random import randint
from sys import stdout

from keras import Model, layers, optimizers, models
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np

from ..pre_processamento.vocabulario import Vocabulario
from .keras_util import CallbackFimEpoca
from .predicao import seleciona_caractere

from .. import configuracao

def modelo_fchollet(vocabulario: Vocabulario) -> Model:
    camadas = [
        layers.Input(shape=(configuracao.tamanho_janela, vocabulario.tamanho)),
        layers.LSTM(128),
        layers.Dense(vocabulario.tamanho, activation='softmax')
    ]

    modelo = models.Sequential(camadas)

    otimizador = optimizers.RMSprop(learning_rate=configuracao.taxa_aprendizagem)
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

    for temperatura in [0.7, 1.0, 1.3]:
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

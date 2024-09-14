from random import randint
from sys import stdout
from typing import Optional

from keras import layers, optimizers, models
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np

from . import configuracao
from .pre_processamento.janelas_one_hot import JanelasOneHot
from .pre_processamento.vocabulario import Vocabulario
from .rede_neural.keras_util import alocar_memoria_aos_poucos, CallbackFimEpoca


def gerar_amostra():
    # índice do primeiro caractere da última janela (começo da última janela)
    indice_comeco_ultima_janela = len(texto_completo) - configuracao.tamanho_janela # -1??

    inicio_janela = randint(0, indice_comeco_ultima_janela)
    fim_janela = inicio_janela + configuracao.tamanho_janela
    janela_aleatoria = texto_completo[inicio_janela : fim_janela]

    print(f'--- Gerando com o seguinte texto inicial: "{janela_aleatoria}"')

    for temperatura in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperatura:', temperatura)
        stdout.write(janela_aleatoria)

        # texto a ser construído, iniciando com a janela aleatória selecionada
        texto_gerado = janela_aleatoria

        # gerando 400 caracteres seguintes
        for i in range(400):
            sampled = np.zeros((1, configuracao.tamanho_janela, vocabulario.tamanho))
            for i, char in enumerate(texto_gerado):
                sampled[0, i, vocabulario.obtem_indice[char]] = 1.

            previsto = model.predict(sampled, verbose=0)[0]

            proximo_indice = seleciona_caractere_predito(previsto, temperatura)
            proximo_caractere = vocabulario.obtem_caractere[proximo_indice]

            texto_gerado += proximo_caractere
            texto_gerado = texto_gerado[1:]

            stdout.write(proximo_caractere)
            stdout.flush()

        print()

def seleciona_caractere_predito(logits: np.ndarray[np.float32], temperatura: float = 1.0) -> np.int64:
    # converte a lista de predições para um array float64
    logits_64 = np.asarray(logits).astype('float64')

    # aplica o logaritmo natural às predições e divide pela temperatura
    logits_temperatura = np.log(logits_64) / temperatura

    # calcula o exponencial das predições
    logits_exp = np.exp(logits_temperatura)

    # normaliza as predições exponenciais de forma que a soma seja 1
    logits_normalizados = logits_exp / np.sum(logits_exp)

    # gera uma amostra a partir de uma distribuição multinomial baseada nas probabilidades calculadas
    probabilidades = np.random.multinomial(1, logits_normalizados, 1)

    # retorna o índice da predição com a maior probabilidade
    return np.argmax(probabilidades)

if __name__ == '__main__':
    alocar_memoria_aos_poucos()

    with open(configuracao.arquivo_txt, encoding='utf-8') as f:
        texto_completo = f.read().lower()

    vocabulario = Vocabulario(texto_completo)
    janelas = JanelasOneHot(texto_completo, vocabulario)

    layers = [
        layers.Input(shape=(configuracao.tamanho_janela, vocabulario.tamanho)),
        layers.LSTM(128),
        layers.Dense(vocabulario.tamanho, activation='softmax')
    ]

    model = models.Sequential(layers)

    optimizer = optimizers.RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    callback_checkpoint = ModelCheckpoint(filepath='modelos_treinados/epoch-{epoch}-loss-{loss}.keras', monitor='loss', save_best_only=True)
    callback_amostra = CallbackFimEpoca(gerar_amostra, 5)
    callback_parada = EarlyStopping(monitor='loss', patience=100, verbose=1)
    callbacks = [callback_checkpoint, callback_amostra, callback_parada]

    model.fit(janelas.x, janelas.y, batch_size=128, epochs=999999, callbacks=callbacks)

from keras.models import Model
from typing import Any, Generator
import numpy as np

from . import configuracao
from .vocabulario import Vocabulario


def gera_proximo_caractere(modelo: Model, vocabulario: Vocabulario, texto_anterior: str,
    temperatura: float) -> str:

    texto_anterior_one_hot = np.zeros((1, configuracao.tamanho_janela, vocabulario.tamanho))
    zeros_esquerda = configuracao.tamanho_janela - len(texto_anterior)

    for i, char in enumerate(texto_anterior):
        texto_anterior_one_hot[0, i + zeros_esquerda, vocabulario.obtem_indice[char]] = 1.

    previsto = modelo.predict(texto_anterior_one_hot, verbose=0)[0]

    proximo_indice = seleciona_caractere(previsto, temperatura)
    proximo_caractere = vocabulario.obtem_caractere[int(proximo_indice)]

    return proximo_caractere


def gera_proximo_caractere_continuamente(modelo: Model, vocabulario: Vocabulario, texto_anterior: str,
    temperatura: float) -> Generator[str, None, None]:

    while True:

        # realiza o forward pass, aplica a temperatura, e obtém o próximo caractere previsto
        proximo_caractere = gera_proximo_caractere(modelo, vocabulario, texto_anterior, temperatura)

        # retornando o caractere obtido
        yield proximo_caractere

        # remove primeiro caractere
        texto_anterior = texto_anterior[1:]

        # adiciona o novo caractere
        texto_anterior += proximo_caractere


def seleciona_caractere(logits: np.ndarray[Any, np.dtype[np.float32]], temperatura: float = 1.0) -> np.int64:
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

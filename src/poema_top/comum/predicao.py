'''
Módulo que expõe a lógica de previsão de caractere do modelo (forward pass e pós-processamento).
'''

from typing import Any, Generator

from keras.models import Model
from numpy import argmax, asarray, dtype, exp, float32, int64, log, ndarray, zeros
from numpy import sum as npsum
from numpy.random import multinomial

from . import configuracao
from .vocabulario import Vocabulario

def gera_proximo_caractere(modelo: Model, vocabulario: Vocabulario, texto_anterior: str,
    temperatura: float) -> str:
    '''
    Realiza um forward pass no modelo carregado passado, aplica temperatura, e retorna o caractere previsto pelo modelo.
    '''

    texto_anterior_one_hot = zeros((1, configuracao.tamanho_janela, vocabulario.tamanho))
    zeros_esquerda = configuracao.tamanho_janela - len(texto_anterior)

    for i, char in enumerate(texto_anterior):
        texto_anterior_one_hot[0, i + zeros_esquerda, vocabulario.obtem_indice[char]] = 1.

    previsto = modelo.predict(texto_anterior_one_hot, verbose=0)[0]

    proximo_indice = seleciona_caractere(previsto, temperatura)
    proximo_caractere = vocabulario.obtem_caractere[int(proximo_indice)]

    return proximo_caractere

def gera_proximo_caractere_continuamente(modelo: Model, vocabulario: Vocabulario, texto_anterior: str,
    temperatura: float) -> Generator[str, None, None]:
    '''
    Retorna um gerador que a cada iteração: realiza um forward pass no modelo carregado passado, aplica temperatura, e
    retorna o caractere previsto pelo modelo,
    '''

    while True:

        # realiza o forward pass, aplica a temperatura, e obtém o próximo caractere previsto
        proximo_caractere = gera_proximo_caractere(modelo, vocabulario, texto_anterior, temperatura)

        # retornando o caractere obtido
        yield proximo_caractere

        # remove primeiro caractere
        texto_anterior = texto_anterior[1:]

        # adiciona o novo caractere
        texto_anterior += proximo_caractere

def seleciona_caractere(probabilidades: ndarray[Any, dtype[float32]], temperatura: float = 1.0) -> int64:
    '''
    Recebe as probabilidades e faz a seleção do próximo caractere, de acordo com a temperatura. A função recebe
    probabilidades já normalizadas pela função softmax, e não os logits 'puros'.
    '''

    # converte a lista de predições para um array float64
    probabilidades_64 = asarray(probabilidades).astype('float64')

    # aplica o logaritmo natural às predições e divide pela temperatura
    probabilidades_temperatura = log(probabilidades_64) / temperatura

    # calcula o exponencial das predições
    probabilidades_exponencial = exp(probabilidades_temperatura)

    # normaliza as predições exponenciais de forma que a soma seja 1
    probabilidades_normalizadas = probabilidades_exponencial / npsum(probabilidades_exponencial)

    # gera uma amostra a partir de uma distribuição multinomial baseada nas probabilidades calculadas
    probabilidades = multinomial(1, probabilidades_normalizadas, 1).astype(float)

    # retorna o índice da predição com a maior probabilidade
    return argmax(probabilidades)

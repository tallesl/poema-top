'''
Módulo que auxilia na manipulação do dataset de texto, quebrando em janelas e codificando em one-hot.
'''

from typing import Any

from numpy import dtype, float32, ndarray, zeros

from . import configuracao # pylint: disable=no-name-in-module
from ..comum import configuracao as configuracao_comum
from ..comum.vocabulario import Vocabulario

class JanelasOneHot:
    '''
    Classe que recebe o texto completo do dataset e o vocabulário, quebra em janelas, codifica em one-hot, e deixa
    preparada o x e y a ser utilizado no treino (sendo x o texto anterior em one-hot, e y o próximo caractere em
    one-hot.
    '''
    def __init__(self, texto_completo: str, vocabulario: Vocabulario) -> None:
        assert texto_completo
        assert vocabulario
        assert len(texto_completo) > configuracao_comum.tamanho_janela

        janelas_exceto_ultimo, janelas_ultimo = _quebra_em_janelas(texto_completo)
        self.x, self.y = _one_hot_janelas(janelas_exceto_ultimo, janelas_ultimo, vocabulario)

def _quebra_em_janelas(texto_completo: str) -> tuple[list[str], list[str]]:
    '''
    Quebra o texto passado em janelas e retorna uma tupla com as janelas e o próximo caractere da janela.
    '''

    assert texto_completo

    total_caracteres = len(texto_completo)

    janelas_exceto_ultimo = [] # janelas de texto (x) sem o último caractere (y)
    janelas_ultimo = [] # último caractere das janelas (y)

    # iterando do primeiro caractere da primeira janela (zero) até o primeiro caractere da última janela
    # a cada iteração pulamos a distância configurada
    for i in range(0, total_caracteres, configuracao.distancia_janela):

        inicio_janela = i
        fim_janela = i + configuracao_comum.tamanho_janela - 1

        # a última janela é descartada para garantir que as janelas tenham o mesmo tamanho
        if fim_janela >= total_caracteres:
            break

        exceto_ultimo = texto_completo[inicio_janela:fim_janela] # janelas de caracteres exceto o último
        ultimo = texto_completo[fim_janela] # último caractere

        # exceto último + último = janela completa
        assert len(exceto_ultimo) + len(ultimo) == configuracao_comum.tamanho_janela

        janelas_exceto_ultimo.append(exceto_ultimo)
        janelas_ultimo.append(ultimo)

    # janelas_exceto_ultimo e janelas_ultimo devem possuir índices que casam e devem ter o mesmo tamanho
    assert len(janelas_exceto_ultimo) == len(janelas_ultimo)

    return janelas_exceto_ultimo, janelas_ultimo

def _one_hot_janelas(janelas_exceto_ultimo: list[str], janelas_ultimo: list[str],
    vocabulario: Vocabulario) -> tuple[ndarray[Any, dtype[float32]], ndarray[Any, dtype[float32]]]:
    '''
    Codifica as janelas e os últimos caracteres recebidos em one-hot.
    '''

    assert janelas_exceto_ultimo
    assert janelas_ultimo
    assert vocabulario
    assert len(janelas_exceto_ultimo) == len(janelas_ultimo)

    total_janelas = len(janelas_exceto_ultimo)

    # "x" são as janelas sem o último caracteres codificadas em one hot
    # shape: (total janelas, quantidade de caracteres da janela - 1, tamanho do vocabulario)

    # "y" são os últimos caracteres (de cada janela) codificados em one hot
    # shape: (total janelas, tamanho do vocabulário)

    shape_x = (total_janelas, configuracao_comum.tamanho_janela - 1, vocabulario.tamanho)
    shape_y = (total_janelas, vocabulario.tamanho)

    x = zeros(shape_x, dtype=bool)
    y = zeros(shape_y, dtype=bool)

    # para cada janela
    for i in range(total_janelas):

        janela_x = janelas_exceto_ultimo[i]
        caractere_y = janelas_ultimo[i]

        x[i] = vocabulario.one_hot_texto(janela_x)
        y[i] = vocabulario.one_hot_caractere(caractere_y)

    return x, y

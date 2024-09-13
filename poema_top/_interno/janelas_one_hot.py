import numpy as np

from .. import configuracao
from .vocabulario import Vocabulario

class JanelasOneHot:
    def __init__(self, texto_completo: str, vocabulario: Vocabulario) -> None:
        assert texto_completo
        assert vocabulario

        janelas_menos_ultimo, janelas_ultimo = _janelas(texto_completo)
        self.x, self.y = _one_hot(janelas_menos_ultimo, janelas_ultimo, vocabulario)
        import pudb;pu.db

def _janelas(texto_completo: str) -> tuple[list[str], list[str]]:
    assert texto_completo

    janelas_menos_ultimo = [] # janelas de texto (x) sem o último caractere (y)
    janelas_ultimo = [] # último caractere das janelas (y)

    # índice do primeiro caractere da última janela (começo da última janela)
    indice_comeco_ultima_janela = len(texto_completo) - configuracao.tamanho_janela

    # iterando do primeiro caractere da primeira janela (zero) até o primeiro caractere da última janela
    # a cada iteração pulamos a distância configurada
    for i in range(0, indice_comeco_ultima_janela, configuracao.distancia_janela):

        exceto_ultimo = texto_completo[i:i + configuracao.tamanho_janela] # window characters minus last one
        ultimo = texto_completo[i + configuracao.tamanho_janela] # window last character

        janelas_menos_ultimo.append(exceto_ultimo)
        janelas_ultimo.append(ultimo)

    # windows and windows_last have matching indexes and the same length
    return janelas_menos_ultimo, janelas_ultimo

def _one_hot(janelas_menos_ultimo: list[str], janelas_ultimo: list[str], vocabulario: Vocabulario) -> tuple[
    np.ndarray, np.ndarray]:

    assert janelas_menos_ultimo
    assert janelas_ultimo
    assert vocabulario
    assert len(janelas_menos_ultimo) == len(janelas_ultimo)

    total_janelas = len(janelas_menos_ultimo)

    x = np.zeros((total_janelas, configuracao.tamanho_janela, vocabulario.tamanho), dtype=bool)
    y = np.zeros((total_janelas, vocabulario.tamanho), dtype=bool)

    # para cada janela
    for i, janela_x in enumerate(janelas_menos_ultimo):

        # para cada caractere na janela
        for j, caractere_x in enumerate(janela_x):

            # obtendo índice do caractere (x)
            k = vocabulario.obtem_indice[caractere_x]

            # "i" indica onde nas janelas, "j" indica onde na janela atual, "k" qual o índice do caractere atual
            x[i, j, k] = 1

        # obtendo índice do caractere (y)
        caractere_y = janelas_ultimo[i]
        j = vocabulario.obtem_indice[caractere_y]

        # "i" indica onde nos últimos caracters, "j" qual o índice do caractere atual
        y[i, j] = 1

    # "x" são as janelas sem o último caracteres codificadas em one hot
    # shape: (total janelas, quantidade de caracteres da janela, tamanho do vocabulario)

    # "y" são os últimos caracteres codificados em one hot
    # shape: (total janelas, tamanho do vocabulário)
    return x, y

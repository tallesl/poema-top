import numpy as np

from .. import configuracao
from .vocabulario import Vocabulario

class JanelasOneHot:
    def __init__(self, texto_completo: str, vocabulario: Vocabulario) -> None:
        assert texto_completo
        assert vocabulario

        janelas_menos_ultimo, janelas_ultimo = _quebra_em_janelas(texto_completo)
        self.x, self.y = _one_hot_janelas(janelas_menos_ultimo, janelas_ultimo, vocabulario)

def _quebra_em_janelas(texto_completo: str) -> tuple[list[str], list[str]]:
    assert texto_completo

    total_caracteres = len(texto_completo)

    janelas_menos_ultimo = [] # janelas de texto (x) sem o último caractere (y)
    janelas_ultimo = [] # último caractere das janelas (y)

    # iterando do primeiro caractere da primeira janela (zero) até o primeiro caractere da última janela
    # a cada iteração pulamos a distância configurada
    for i in range(0, total_caracteres, configuracao.distancia_janela):

        inicio_janela = i
        fim_janela = i + configuracao.tamanho_janela - 1

        # a última janela é descartada para garantir que as janelas tenham o mesmo tamanho
        if fim_janela >= total_caracteres:
            break

        exceto_ultimo = texto_completo[inicio_janela:fim_janela] # janelas de caracteres menos o último
        ultimo = texto_completo[fim_janela] # último caractere

        # exceto último + último = janela completa 
        assert len(exceto_ultimo) + len(ultimo) == configuracao.tamanho_janela

        janelas_menos_ultimo.append(exceto_ultimo)
        janelas_ultimo.append(ultimo)

    # janelas_menos_ultimo e janelas_ultimo devem possuir índices que casam e devem ter o mesmo tamanho
    assert len(janelas_menos_ultimo) == len(janelas_ultimo)

    return janelas_menos_ultimo, janelas_ultimo

def _one_hot_janelas(janelas_menos_ultimo: list[str], janelas_ultimo: list[str], vocabulario: Vocabulario) -> tuple[
    np.ndarray, np.ndarray]:

    assert janelas_menos_ultimo
    assert janelas_ultimo
    assert vocabulario
    assert len(janelas_menos_ultimo) == len(janelas_ultimo)

    total_janelas = len(janelas_menos_ultimo)

    # "x" são as janelas sem o último caracteres codificadas em one hot
    # shape: (total janelas, quantidade de caracteres da janela - 1, tamanho do vocabulario)

    # "y" são os últimos caracteres (de cada janela) codificados em one hot
    # shape: (total janelas, tamanho do vocabulário)

    shape_x = (total_janelas, configuracao.tamanho_janela - 1, vocabulario.tamanho)
    shape_y = (total_janelas, vocabulario.tamanho)

    x = np.zeros(shape_x, dtype=bool)
    y = np.zeros(shape_y, dtype=bool)

    # para cada janela
    for i in range(total_janelas):

        janela_x = janelas_menos_ultimo[i]
        caractere_y = janelas_ultimo[i]

        x[i] = vocabulario.one_hot_texto(janela_x)
        y[i] = vocabulario.one_hot_caractere(caractere_y)

    return x, y

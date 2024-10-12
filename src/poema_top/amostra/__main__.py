from itertools import count, islice
from random import randint
from sys import stdout
from typing import Optional

from keras.models import Model
import numpy as np

from . import configuracao
from ..comum import configuracao as configuracao_comum
from ..comum.dataset import le_txt_dataset, obtem_janela_aleatoria
from ..comum.keras import alocar_memoria_aos_poucos, carrega_ultimo_modelo
from ..comum.log import LogaMemoria
from ..comum.predicao import gera_proximo_caractere_continuamente
from ..comum.vocabulario import Vocabulario


def main():
    try:
        alocar_memoria_aos_poucos()

        print('Lendo txt...')
        texto_completo = le_txt_dataset()

        print('Montando vocabulário...')
        vocabulario = Vocabulario(texto_completo)

        print('Carregando último modelo...')
        modelo = carrega_ultimo_modelo()

        print('Obtendo uma janela aleatória do texto lido...')
        janela_aleatoria = obtem_janela_aleatoria(texto_completo)

        print('Gerando amostras...')

        for temperatura in configuracao.temperatura_amostras:
            print(f'\n\n---------------------------\n\nAmostra com temperatura {temperatura}:\n\n[{janela_aleatoria}]',
                end='', flush=True)

            gerador_caracteres_infinito = gera_proximo_caractere_continuamente(modelo, vocabulario, janela_aleatoria,
                temperatura)

            gerador_caracteres = islice(gerador_caracteres_infinito, configuracao.tamanho_amostra)

            for caractere in gerador_caracteres:
                print(caractere, end='', flush=True)

    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()

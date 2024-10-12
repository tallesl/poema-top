from random import randint
from sys import stdout

from keras.models import Model
import numpy as np

from . import configuracao
from ..comum import configuracao as configuracao_comum
from ..comum.dataset import le_txt_dataset
from ..comum.keras import alocar_memoria_aos_poucos, carrega_ultimo_modelo
from ..comum.log import LogaMemoria
from ..comum.predicao import gera_proximo_caractere, seleciona_caractere
from ..comum.vocabulario import Vocabulario


def gera_amostras(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> None:

    def print_separador():
        print()
        print()
        print('---------------------------')
        print()

    indice_comeco_ultima_janela = len(texto_completo) - configuracao_comum.tamanho_janela - 1
    inicio_janela = randint(0, indice_comeco_ultima_janela)
    fim_janela = inicio_janela + configuracao_comum.tamanho_janela

    janela_aleatoria = texto_completo[inicio_janela : fim_janela]

    print_separador()

    for temperatura in configuracao.temperatura_amostras:
        print(f'Amostra com temperatura {temperatura}:')
        print()
        print(f'[{janela_aleatoria}]', end='', flush=True)

        janela_atual = janela_aleatoria

        for i in range(configuracao.tamanho_amostras):

            proximo_caractere = gera_proximo_caractere(modelo, vocabulario, janela_atual, temperatura)

            # remove primeiro caractere
            janela_atual = janela_atual[1:]

            # adiciona novo caractere predito
            janela_atual += proximo_caractere

            stdout.write(proximo_caractere)
            stdout.flush()

        print_separador()


def main():
    alocar_memoria_aos_poucos()

    print('Lendo txt...')
    texto_completo = le_txt_dataset()

    print('Montando vocabulário...')
    vocabulario = Vocabulario(texto_completo)

    print('Carregando último modelo...')
    modelo = carrega_ultimo_modelo()

    print('Gerando amostras...')
    gera_amostras(modelo, vocabulario, texto_completo)


if __name__ == '__main__':
    main()

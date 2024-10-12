from random import randint
from sys import stdout

from keras.models import Model
import numpy as np

from ..comum.dataset import le_txt_dataset
from ..comum.keras import alocar_memoria_aos_poucos, carrega_ultimo_modelo
from ..comum.log import LogaMemoria
from ..comum.predicao import gera_proximo_caractere, seleciona_caractere
from ..comum.vocabulario import Vocabulario


def insere_caractere(texto_atual: str, proximo_caractere: str) -> str:
        # remove primeiro caractere
        proximo_texto = texto_atual[1:]

        # adiciona novo caractere predito
        proximo_texto += proximo_caractere

        return proximo_texto


def main():
    try:
        alocar_memoria_aos_poucos()

        print('Lendo txt...')
        texto_completo = le_txt_dataset()

        print('Montando vocabulário...')
        vocabulario = Vocabulario(texto_completo)

        print('Carregando último modelo...')
        modelo = carrega_ultimo_modelo()

        print('Temperatura: ', end='', flush=True)
        temperatura = float(input())

        print('Texto inicial: ', end='', flush=True)
        texto_atual = input().lower()

        while True:

            proximo_caractere = gera_proximo_caractere(modelo, vocabulario, texto_atual, temperatura)
            texto_atual = insere_caractere(texto_atual, proximo_caractere)

            stdout.write(proximo_caractere)
            stdout.flush()

    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()

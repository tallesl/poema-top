from os import listdir, path
from random import randint
from sys import stdout

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM
from keras.models import load_model, Model, Sequential
from keras.optimizers import RMSprop
import numpy as np

from .. import configuracao
from ..pre_processamento.vocabulario import Vocabulario
from .keras_util import CallbackFimEpoca
from .predicao import seleciona_caractere

from .. import configuracao

def cria_modelo(vocabulario: Vocabulario) -> Model:
    # "a plain stack of layers where each layer has exactly one input tensor and one output tensor"
    modelo = Sequential()

    # camada de entrada
    formato_entrada = (configuracao.tamanho_janela, vocabulario.tamanho)
    modelo.add(Input(shape=formato_entrada))

    # camada LSTM opcional
    if configuracao.duas_camadas_lstm:
        modelo.add(LSTM(128, return_sequences=True))

    # camada LSTM sempre present
    modelo.add(LSTM(128))

    # camada de saída
    modelo.add(Dense(vocabulario.tamanho, activation='softmax'))

    # atualiza os pesos a cada batch utilizando a média móvel dos quadrados dos gradientes das últimas N iterações,
    # ajustando as taxas de aprendizado (taxa individual para cada peso), prevenindo a explosão e dissipação dos
    # gradientes
    otimizador = RMSprop(learning_rate=configuracao.taxa_aprendizagem)
    modelo.compile(loss='categorical_crossentropy', optimizer=otimizador)

    return modelo


def carrega_ultimo_modelo() -> Model:

    # listando todos arquivos .keras do diretório
    diretorio = path.split(configuracao.caminho_checkpoint)[0]
    arquivos = listdir(diretorio)
    arquivos_keras = [path.join(diretorio, a) for a in arquivos if a.endswith('.keras')]
    
    # se nenhum arquivo .keras foi encontrado
    if not arquivos_keras:
        raise FileNotFoundError(f"No .keras model files found in the directory: {folder_path}")
    
    # obtém o arquivo com a data de alteração mais recente
    ultimo_modelo = max(arquivos_keras, key=path.getmtime)
    
    # carrega o modelo
    modelo = load_model(ultimo_modelo)

    print(f'Carregado modelo "{ultimo_modelo}".')

    return modelo


def callbacks_treino(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> list[Callback]:
    lambda_amostras = lambda: gera_amostras(modelo, vocabulario, texto_completo)

    callback_checkpoint = ModelCheckpoint(filepath=configuracao.caminho_checkpoint, monitor='loss', save_best_only=True)
    callback_amostra = CallbackFimEpoca(lambda_amostras, configuracao.epocas_amostras)
    callback_parada = EarlyStopping(monitor='loss', patience=configuracao.paciencia_treino, verbose=1)

    return [callback_checkpoint, callback_amostra, callback_parada]


def gera_amostras(modelo: Model, vocabulario: Vocabulario, texto_completo: str) -> None:

    def print_separador():
        print()
        print()
        print('---------------------------')
        print()

    indice_comeco_ultima_janela = len(texto_completo) - configuracao.tamanho_janela - 1
    inicio_janela = randint(0, indice_comeco_ultima_janela)
    fim_janela = inicio_janela + configuracao.tamanho_janela

    janela_aleatoria = texto_completo[inicio_janela : fim_janela]

    print_separador()

    for temperatura in configuracao.temperatura_amostras:
        print(f'Amostra com temperatura {temperatura}:')
        print()
        print(f'[{janela_aleatoria}]', end='', flush=True)

        janela_atual = janela_aleatoria

        for i in range(configuracao.tamanho_amostras):
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

        print_separador()

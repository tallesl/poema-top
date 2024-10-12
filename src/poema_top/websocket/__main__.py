from asyncio import get_event_loop, sleep, gather, create_task
from websockets import serve, ConnectionClosed
from os import listdir, path
from random import randint
from sys import stdout
from keras.models import load_model, Model
import numpy as np

from . import configuracao
from ..comum import configuracao as configuracao_comum
from ..comum.dataset import le_txt_dataset
from ..comum.keras import alocar_memoria_aos_poucos
from ..comum.predicao import gera_proximo_caractere, seleciona_caractere
from ..comum.log import LogaMemoria
from ..comum.vocabulario import Vocabulario

# Variáveis globais
modelo = None
vocabulario = None
texto_completo = None
conexoes = []  # Lista de conexões ativas

async def inferencia_continua() -> None:
    """
    Função de inferência contínua que roda indefinidamente e gera texto,
    transmitindo o texto gerado para todas as conexões ativas.
    """
    global conexoes

    indice_comeco_ultima_janela = len(texto_completo) - configuracao_comum.tamanho_janela - 1
    inicio_janela = randint(0, indice_comeco_ultima_janela)
    fim_janela = inicio_janela + configuracao_comum.tamanho_janela
    janela_atual = texto_completo[inicio_janela : fim_janela]
    temperatura = configuracao.temperatura

    try:
        while True:
            # Gera um novo caractere
            proximo_caractere = gera_proximo_caractere(modelo, vocabulario, janela_atual, temperatura)

            # Atualiza a janela
            janela_atual = janela_atual[1:] + proximo_caractere

            # Envia o caractere gerado para todas as conexões ativas
            if conexoes:
                await gather(*[websocket.send(proximo_caractere) for websocket in conexoes])

            # Espera um curto período antes de gerar o próximo caractere
            await sleep(0.1)  # Ajuste o tempo de acordo com a velocidade desejada

    except Exception as e:
        print(f'Erro na inferência contínua: {e}')

async def handler(websocket, path):
    """
    Handler para cada nova conexão WebSocket. Cada cliente que se conecta
    será adicionado à lista de conexões ativas e receberá os resultados da
    inferência contínua.
    """
    global conexoes
    conexoes.append(websocket)
    print(f'Cliente conectado. Conexões ativas: {len(conexoes)}')

    try:
        # Aguarda até que a conexão seja fechada
        await websocket.wait_closed()
    finally:
        # Remove a conexão da lista de conexões ativas ao desconectar
        conexoes.remove(websocket)
        print(f'Cliente desconectado. Conexões ativas: {len(conexoes)}')

async def inicializa_servidor():
    global modelo, vocabulario, texto_completo

    alocar_memoria_aos_poucos()

    with LogaMemoria('Lendo txt...'):
        texto_completo = le_txt_dataset()

    with LogaMemoria('Montando vocabulário...'):
        vocabulario = Vocabulario(texto_completo)

    with LogaMemoria('Carregando modelo...'):
        modelo = load_model(configuracao.modelo)

    # Inicia a inferência contínua em segundo plano
    create_task(inferencia_continua())

    # Inicializa o servidor WebSocket
    servidor = await serve(handler, "localhost", configuracao.porta)
    print('WebSocket iniciado...')

    await servidor.wait_closed()  # Aguarda até o servidor ser encerrado

def main():
    loop = get_event_loop()
    loop.run_until_complete(inicializa_servidor())
    loop.run_forever()

if __name__ == '__main__':
    main()


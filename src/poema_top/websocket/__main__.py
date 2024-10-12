# type: ignore

from asyncio import get_event_loop, sleep, gather, create_task
from os import listdir, path
from random import randint
from sys import stdout

from keras.models import load_model, Model
from websockets import serve, ConnectionClosed
import numpy as np

from . import configuracao
from ..comum import configuracao as configuracao_comum
from ..comum.dataset import le_txt_dataset, obtem_janela_aleatoria
from ..comum.keras import alocar_memoria_aos_poucos
from ..comum.predicao import gera_proximo_caractere_continuamente
from ..comum.log import LogaMemoria
from ..comum.vocabulario import Vocabulario

modelo = None
vocabulario = None
texto_completo = None
conexoes = []

async def inferencia_continua():
    global conexoes

    texto_inicial = obtem_janela_aleatoria(texto_completo)
    gerador_caracteres = gera_proximo_caractere_continuamente(modelo, vocabulario, texto_inicial,
        configuracao.temperatura)

    while True:
        try:
            proximo_caractere = next(gerador_caracteres)

            if conexoes:
                await gather(*[websocket.send(proximo_caractere) for websocket in conexoes])

            await sleep(0)

        except Exception as e:
            print(e)

async def handler(websocket, path):
    global conexoes
    conexoes.append(websocket)
    print(f'Cliente conectado. Conexões ativas: {len(conexoes)}')

    try:
        await websocket.wait_closed()
    finally:
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

    create_task(inferencia_continua())

    servidor = await serve(handler, 'localhost', configuracao.porta)
    print('WebSocket iniciado...')

    await servidor.wait_closed()

def main():
    loop = get_event_loop()
    loop.run_until_complete(inicializa_servidor())
    loop.run_forever()

if __name__ == '__main__':
    main()

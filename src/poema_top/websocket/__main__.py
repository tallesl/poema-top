# type: ignore

'''
Script que inicia um servidor websocket e serve em tempo real os caracteres sendo previstos pelo modelo em todos os
websockets conectados.
'''

from asyncio import get_event_loop, sleep, gather, create_task

from keras.models import load_model
from websockets import serve

from . import configuracao
from ..comum.dataset import le_txt_dataset, obtem_janela_aleatoria
from ..comum.keras import alocar_memoria_aos_poucos
from ..comum.predicao import gera_proximo_caractere_continuamente
from ..comum.vocabulario import Vocabulario

modelo = None # pylint: disable=invalid-name
vocabulario = None # pylint: disable=invalid-name
texto_completo = None # pylint: disable=invalid-name
conexoes = []

async def inferencia_continua():
    '''
    Realiza inferência continuamente e envia os caracteres gerados em tempo real à todas a conexões ativas.
    '''
    texto_inicial = obtem_janela_aleatoria(texto_completo)
    gerador_caracteres = gera_proximo_caractere_continuamente(modelo, vocabulario, texto_inicial,
        configuracao.temperatura)

    while True:
        try:

            proximo_caractere = next(gerador_caracteres)

            print(proximo_caractere, end='', flush=True)

            if conexoes:
                await gather(*[websocket.send(proximo_caractere) for websocket in conexoes])

            await sleep(0)

        except Exception as e: # pylint: disable=broad-exception-caught
            print(e)

async def handler(websocket, _):
    '''
    Handler de nova conexões.
    '''

    conexoes.append(websocket)
    print(f'\nCliente conectado. Conexões ativas: {len(conexoes)}')

    try:
        await websocket.wait_closed()

    finally:
        conexoes.remove(websocket)
        print(f'\nCliente desconectado. Conexões ativas: {len(conexoes)}')

async def inicializa_servidor():
    '''
    Carrega o dataset e o modelo, e inicializa o servidor websocket.
    '''

    global modelo, vocabulario, texto_completo # pylint: disable=global-statement

    alocar_memoria_aos_poucos()

    print('Lendo txt...')
    texto_completo = le_txt_dataset()

    print('Montando vocabulário...')
    vocabulario = Vocabulario(texto_completo)

    print('Carregando modelo...')
    modelo = load_model(configuracao.modelo)

    print('Iniciado servidor WebSocket...')
    create_task(inferencia_continua())
    servidor = await serve(handler, 'localhost', configuracao.porta)

    print('Servidor iniciado.')
    await servidor.wait_closed()

def main():
    '''
    Função principal da aplicação.
    '''

    # pylint: disable=duplicate-code

    loop = get_event_loop()
    loop.run_until_complete(inicializa_servidor())
    loop.run_forever()

if __name__ == '__main__':
    main()

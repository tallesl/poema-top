from random import randint

from .. import configuracao

def le_txt_dataset() -> str:
    with open(configuracao.dataset_txt, encoding='utf-8') as arquivo:
        texto_completo = arquivo.read()

    if configuracao.converte_lowercase:
        texto_completo = texto_completo.lower()

    if configuracao.remove_terminador:
        texto_completo = texto_completo.replace(configuracao.remove_terminador, '')

    return texto_completo


def extrai_janela(texto_completo: str) -> str:
    tamanho_arquivo = len(texto_completo)
    assert tamanho_arquivo > configuracao.tamanho_janela

    inicio_janela = randint(0, tamanho_arquivo - configuracao.tamanho_janela)
    fim_janela = inicio_janela + configuracao.tamanho_janela

    return texto_completo[inicio_janela : fim_janela]

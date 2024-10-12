'''
Módulo com funções relacionadas à manipulação do dataset (arquivo .txt).
'''

from random import randint

from . import configuracao

def le_txt_dataset() -> str:
    '''
    Lê todo o dataset para memória, converte caracteres para lowercase, remove terminador, e retorna o texto
    transformado.
    '''
    with open(configuracao.dataset_txt, encoding='utf-8') as arquivo:
        texto_completo = arquivo.read()

    if configuracao.converte_lowercase:
        texto_completo = texto_completo.lower()

    if configuracao.remove_terminador:
        texto_completo = texto_completo.replace(configuracao.remove_terminador, '')

    return texto_completo

def obtem_janela_aleatoria(texto_completo: str) -> str:
    '''
    Extrai uma janela de texto do texto completo passado. A janela é retirada de uma parte aleatório do texto e respeita
    o tamanho encontrado no arquivo de configuração.
    '''

    # tamanho total do arquivo
    tamanho_arquivo = len(texto_completo)

    # garantindo que o texto total comporta pelo menos uma janela
    assert tamanho_arquivo > configuracao.tamanho_janela

    # calcula qual o índice do primeiro caractere da última janela do texto completo
    indice_comeco_ultima_janela = tamanho_arquivo - configuracao.tamanho_janela - 1

    # randomiza um valor entre o início do texto até o início da última janela, que foi calculado acima
    inicio_janela = randint(0, indice_comeco_ultima_janela)

    # cálculo simples do índice final da janela baseado no índice inicial obtido acima
    fim_janela = inicio_janela + configuracao.tamanho_janela

    # extrai a janela do texto completo e retorna
    return texto_completo[inicio_janela : fim_janela]

'''
Script que carrega o dataset (arquivo .txt) e o último modelo treinado do diretório de checkpoint, retira uma janela
aleatória de texto do dataset, e realiza o forward pass no modelo com a janela selecionada, em diferentes temperaturas,
exibindo as amostras na linha de comando.
'''

from itertools import islice

from . import configuracao
from ..comum.dataset import le_txt_dataset, obtem_janela_aleatoria
from ..comum.keras import alocar_memoria_aos_poucos, carrega_ultimo_modelo
from ..comum.predicao import gera_proximo_caractere_continuamente
from ..comum.vocabulario import Vocabulario

def main() -> None:
    '''
    Função principal da aplicação.
    '''

    # pylint: disable=duplicate-code

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

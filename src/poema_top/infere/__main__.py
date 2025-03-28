'''
Script que carrega o dataset (arquivo .txt) e o último modelo treinado do diretório de checkpoint, solicita ao usuário
um texto inicial e uma temperatura, e realiza o forward pass continuamente exibindo os caracteres na linha de comando.
'''

from ..comum.dataset import le_txt_dataset
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

        print('Temperatura: ', end='', flush=True)
        temperatura = float(input())

        print('Texto inicial: ', end='', flush=True)
        texto_inicial = input().lower()

        gerador_caracteres = gera_proximo_caractere_continuamente(modelo, vocabulario, texto_inicial, temperatura)

        for caractere in gerador_caracteres:
            print(caractere, end='', flush=True)

    except KeyboardInterrupt:
        print()

if __name__ == '__main__':
    main()

'''
Módulo que expõe callback para plotar o gráfico da perda durante o treinamento.
'''

from typing import Any

from keras.callbacks import Callback
import matplotlib.pyplot as plt

class GraficoLoss(Callback): # type: ignore[misc]
    '''
    Callback customizado do Keras que, a cada época do treinamento, plota no gráfico a perda calculada.
    '''
    def __init__(self, caminho_arquivo: str):
        '''
        Ctor.
        '''

        super().__init__()
        self.caminho_arquivo = caminho_arquivo

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None: # pylint: disable=unused-argument
        '''
        A cada época plota no gráfico a perda calculada.
        '''

        # contando as épocas a partir de 1
        epoch = epoch + 1

        # nada para plotar na primeira época
        if epoch == 1:
            return

        # histórico de treinamento do modelo
        historico = self.model.history.history

        # iniciando uma nova figura para o gráfico
        plt.figure()

        # plota as linha de loss e val_loss
        plt.plot(historico['loss'], label='loss')

        # rótulos dos eixo X e Y
        plt.xlabel('epoch')
        plt.ylabel('loss')

        # legenda para identificando as duas linhas no gráfico
        plt.legend()

        # salva a imagem do gráfico em um arquivo
        plt.savefig(self.caminho_arquivo)

        # fecha a figura para evitar sobreposição de gráficos
        plt.close()

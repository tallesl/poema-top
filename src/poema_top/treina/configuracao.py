# pylint: disable=missing-module-docstring,invalid-name

# ao 'fatiar' o texto em janelas, quantos caracteres são 'pulados' para o ínicio de uma nova janela
distancia_janela = 3

# velocidade de aprendizagem do treino (learning rate)
taxa_aprendizagem = 0.001

# paciência do treino (quantas épocas seguidas com resultado pior que o treino irá suportar antes de encerrar)
paciencia_treino = 50

# quantidade de épocas do treino
epocas_treino = 9999

# caminho dos checkpoints salvos durante treino
diretorio_modelo = '../modelos_treinados/temp/'

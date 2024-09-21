# caminho do arquivo com texto a ser utilizado no treino
dataset_txt = '../dataset_treino/poemas-10k-llama3.1.txt'

# convertendo para lowercase faz o vocabulário ficar menor e diminui consideravelmente a memória ocupada
converte_lowercase = True

# terminador entre os registros do dataset a ser removido
# utilize None para desabilitar a remoção
remove_terminador = '@\n'

# quantidade de caracteres de cada janela de texto
tamanho_janela = 60

# ao 'fatiar' o texto em janelas, quantos caracteres são 'pulados' para o ínicio de uma nova janela
distancia_janela = 3

# velocidade de aprendizagem do treino (learning rate)
taxa_aprendizagem = 0.001

# gerar amostra após quantas épocas
epocas_amostras = 10

# temperatura e quantidade de amostras a ser gerada
temperatura_amostras = [0.3, 0.6, 0.9]

# quantidade de caracteres a ser gerado em cada amostra
tamanho_amostras = 200

# paciência do treino (quantas épocas seguidas com resultado pior que o treino irá suportar antes de encerrar)
paciencia_treino = 50

# quantidade de épocas do treino
epocas_treino = 9999

# caminho dos checkpoints salvos durante treino
caminho_checkpoint = '../modelos_treinados/epoch-{epoch}-loss-{loss}.keras'

# unidades (neurônios) da camada LSTM
unidades_lstm = 128

# se deve-se utilizar apenas uma camada ou duas camadas LSTM, uma seguida outra
duas_camadas_lstm = False

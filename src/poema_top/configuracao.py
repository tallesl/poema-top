# caminho do arquivo com texto a ser utilizado no treino
dataset_txt = '../dataset_treino/poemas-10k.txt'

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
epocas_amostra = 10

# paciência do treino (quantas épocas seguidas com resultado pior que o treino irá suportar antes de encerrar)
paciencia_treino = 50

# quantidade de épocas do treino
epocas_treino = 9999

# caminho dos checkpoints salvos durante treino
caminho_checkpoint = '../modelos_treinados/epoch-{epoch}-loss-{loss}.keras'

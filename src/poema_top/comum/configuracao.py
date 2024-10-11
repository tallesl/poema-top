# caminho do arquivo com texto a ser utilizado no treino
dataset_txt = '../dataset/poemas-10k-llama3.1.txt'

# convertendo para lowercase faz o vocabulário ficar menor e diminui consideravelmente a memória ocupada
converte_lowercase = True

# terminador entre os registros do dataset a ser removido
# utilize None para desabilitar a remoção
remove_terminador = '@\n'

# quantidade de caracteres de cada janela de texto
tamanho_janela = 60

# diretório dos checkpoints salvos durante treino
diretorio_checkpoint = '../modelos_treinados/temp/'

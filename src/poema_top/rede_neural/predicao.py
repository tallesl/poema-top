import numpy as np


def seleciona_caractere(logits: np.ndarray[np.float32], temperatura: float = 1.0) -> np.int64:
    # converte a lista de predições para um array float64
    logits_64 = np.asarray(logits).astype('float64')

    # aplica o logaritmo natural às predições e divide pela temperatura
    logits_temperatura = np.log(logits_64) / temperatura

    # calcula o exponencial das predições
    logits_exp = np.exp(logits_temperatura)

    # normaliza as predições exponenciais de forma que a soma seja 1
    logits_normalizados = logits_exp / np.sum(logits_exp)

    # gera uma amostra a partir de uma distribuição multinomial baseada nas probabilidades calculadas
    probabilidades = np.random.multinomial(1, logits_normalizados, 1)

    # retorna o índice da predição com a maior probabilidade
    return np.argmax(probabilidades)

import numpy as np

from ..configuracao import tamanho_janela

class Vocabulario:
    def __init__(self, texto_completo: str):
        assert texto_completo
        assert len(texto_completo) > tamanho_janela

        # caracteres únicos ordenados
        caracteres = sorted(set(texto_completo))

        # mapeando caractere para índice e vice-versa
        self.obtem_indice = {c: i for i, c in enumerate(caracteres)}
        self.obtem_caractere = {i: c for i, c in enumerate(caracteres)}
        self.tamanho = len(caracteres)


    def one_hot_texto(self, texto: str) -> np.ndarray[np.ndarray[bool]]:
        assert texto

        # convertendo cada caractere do texto para one hot
        array = [self.one_hot_caractere(c) for c in texto]

        # convertendo para um array do NumPy
        array = np.array(array)

        assert array.shape == (len(texto), self.tamanho)
        assert array.dtype == bool

        return array


    def one_hot_caractere(self, caractere: str) -> np.ndarray[bool]:
        assert caractere
        assert len(caractere) == 1

        # criando um array de valores boleanos do tamanho do vocabulário
        # os índices deste array casam com os índices do vocabulário (mesmo caractere no mesmo índice)
        array = np.zeros(self.tamanho, dtype=bool)

        # obtendo o índice do caractere passado
        indice = self.obtem_indice[caractere]

        # trocando de 0 para 1 o índice do caractere passado, no array que foi inicializado com zeros
        array[indice] = 1

        return array

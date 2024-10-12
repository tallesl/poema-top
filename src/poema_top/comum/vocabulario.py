import numpy as np
from numpy.typing import NDArray

class Vocabulario:
    def __init__(self, texto_completo: str):
        assert texto_completo

        # caracteres únicos ordenados
        caracteres = sorted(set(texto_completo))

        # mapeando caractere para índice e vice-versa
        self.obtem_indice = {c: i for i, c in enumerate(caracteres)}
        self.obtem_caractere = dict(enumerate(caracteres))
        self.tamanho = len(caracteres)


    def one_hot_texto(self, texto: str) -> NDArray[np.bool_]:
        assert texto

        # convertendo cada caractere do texto para one hot
        array = [self.one_hot_caractere(c) for c in texto]

        # convertendo para um array do NumPy
        np_array = np.array(array)

        assert np_array.shape == (len(texto), self.tamanho)
        assert np_array.dtype == bool

        return np_array


    def one_hot_caractere(self, caractere: str) -> NDArray[np.bool_]:
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

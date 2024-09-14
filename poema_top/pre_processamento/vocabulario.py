class Vocabulario:
    def __init__(self, texto_completo: str):
        assert texto_completo

        # caracteres únicos ordenados
        caracteres = sorted(set(texto_completo))

        # mapeando caractere para índice e vice-versa
        self.obtem_indice = {c: i for i, c in enumerate(caracteres)}
        self.obtem_caractere = {i: c for i, c in enumerate(caracteres)}
        self.tamanho = len(caracteres)

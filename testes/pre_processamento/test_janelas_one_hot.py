from unittest.mock import patch

import numpy as np

from poema_top.pre_processamento.janelas_one_hot import JanelasOneHot, _quebra_em_janelas, _one_hot_janelas
from poema_top.pre_processamento.vocabulario import Vocabulario


def teste_quebra_em_janelas():
    with patch('poema_top.configuracao.tamanho_janela', 5), \
        patch('poema_top.configuracao.distancia_janela', 3):

        # arrange

        # "hello world" em janelas de tamanho 5 e distância de 3 gera:
        # hell+o
        # lo w+o
        # worl+d
        exceto_ultimo_esperado = ['hell', 'lo w', 'worl']
        ultimo_esperado = ['o', 'o', 'd']

        # act

        # "!!" é descartado pois não completa uma janela inteira
        janelas_exceto_ultimo, janelas_ultimo = _quebra_em_janelas('hello world!!')

        # assert
        assert janelas_exceto_ultimo == exceto_ultimo_esperado
        assert janelas_ultimo == ultimo_esperado


def teste_one_hot_janelas():
    with patch('poema_top.configuracao.tamanho_janela', 3), \
        patch('poema_top.configuracao.distancia_janela', 1):

        # arrange
        vocabulario = Vocabulario('abcd') # "d" será descartado pois não completa uma janela inteira
        one_hot_a = vocabulario.one_hot_caractere('a')
        one_hot_b = vocabulario.one_hot_caractere('b')
        one_hot_c = vocabulario.one_hot_caractere('c')

        janelas_exceto_ultimo = ['ab']
        janelas_ultimo = ['c']

        exceto_ultimo_esperado = np.array([[one_hot_a, one_hot_b]])
        ultimo_esperado = np.array([one_hot_c])

        # act
        one_hot_exceto_ultimo, one_hot_ultimo = _one_hot_janelas(janelas_exceto_ultimo, janelas_ultimo, vocabulario)

        # assert

        # total janelas = 1, tamanho janela - 1 = 2, vocabulário = 4
        assert one_hot_exceto_ultimo.shape == (1, 2, 4)

        # total janelas = 1, vocabulario = 4
        assert one_hot_ultimo.shape == (1, 4)

        np.testing.assert_array_equal(one_hot_exceto_ultimo, exceto_ultimo_esperado)
        np.testing.assert_array_equal(one_hot_ultimo, ultimo_esperado)


def teste_classe():
    with patch('poema_top.configuracao.tamanho_janela', 5), \
        patch('poema_top.configuracao.distancia_janela', 3):

        # arrange
        texto = 'hello world!!'
        vocabulario = Vocabulario(texto)

        # act
        janelas = JanelasOneHot(texto, vocabulario)

        # assert

        # total janelas = 3, tamanho janela - 1 = 4, vocabulário = 9
        assert janelas.x.shape == (3, 4, 9)

        # total janelas = 3, vocabulário = 9
        assert janelas.y.shape == (3, 9)

        assert janelas.x.dtype == bool
        assert janelas.y.dtype == bool

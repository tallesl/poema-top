import numpy as np

from poema_top.pre_processamento.vocabulario import Vocabulario

# vocabulário de 'hello world':
# 0: ' '
# 1: 'd'
# 2: 'e'
# 3: 'h'
# 4: 'l'
# 5: 'o'
# 6: 'r'
# 7: 'w'


def teste_obtem():
    # act
    vocabulario = Vocabulario('hello world')

    # assert
    assert vocabulario.obtem_indice['h'] == 3
    assert vocabulario.obtem_indice['e'] == 2
    assert vocabulario.obtem_indice['l'] == 4
    assert vocabulario.obtem_indice['o'] == 5
    assert vocabulario.obtem_indice[' '] == 0
    assert vocabulario.obtem_indice['w'] == 7
    assert vocabulario.obtem_indice['r'] == 6
    assert vocabulario.obtem_indice['d'] == 1

    assert vocabulario.obtem_caractere[0] == ' '
    assert vocabulario.obtem_caractere[1] == 'd'
    assert vocabulario.obtem_caractere[2] == 'e'
    assert vocabulario.obtem_caractere[3] == 'h'
    assert vocabulario.obtem_caractere[4] == 'l'
    assert vocabulario.obtem_caractere[5] == 'o'
    assert vocabulario.obtem_caractere[6] == 'r'
    assert vocabulario.obtem_caractere[7] == 'w'


def teste_tamanho():
    # act
    vocabulario = Vocabulario('hello world')

    # assert
    assert vocabulario.tamanho == 8


def teste_one_hot_texto():
    # arrange
    vocabulario = Vocabulario('hello world')
    one_hot_l = [False, False, False, False, True, False, False, False]
    one_hot_o = [False, False, False, False, False, True, False, False]
    esperado = np.array([one_hot_l, one_hot_o, one_hot_l])

    # act
    one_hot = vocabulario.one_hot_texto('lol')

    # assert
    np.testing.assert_array_equal(one_hot, esperado)


def teste_one_hot_caractere():
    # arrange
    vocabulario = Vocabulario('hello world')
    esperado = np.array([True, False, False, False, False, False, False, False])

    # act
    one_hot = vocabulario.one_hot_caractere(' ')

    # assert
    np.testing.assert_array_equal(one_hot, esperado)

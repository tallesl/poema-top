from .. import configuracao

def le_txt_dataset() -> str:
    with open(configuracao.dataset_txt, encoding='utf-8') as f:
        texto_completo = f.read()

    if configuracao.remove_terminador:
        texto_completo = texto_completo.replace(configuracao.remove_terminador, '')

    return texto_completo

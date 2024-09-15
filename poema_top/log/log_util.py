from time import time
from ..rede_neural.keras_util import obtem_vram


class LogaMemoria:
    def __init__(self):
        self.tempo_inicio = time()

    def __enter__(self):
        self.ram_inicial = _obtem_ram()
        self.vram_inicial, self.vram_pico_inicial = obtem_vram()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            return

        tempo_decorrido = time() - self.tempo_inicio
        tempo_formatado = _formata_segundos(tempo_decorrido)
        ram_final = _obtem_ram()
        vram_final, vram_pico_final = obtem_vram()

        print(f'• {tempo_formatado} decorridos')
        loga_consumo_memoria((self.ram_inicial, ram_final), (self.vram_inicial, self.vram_pico_inicial), (vram_final, vram_pico_final))


def loga_consumo_memoria(ram: tuple[float, float], vram: tuple[float, float], vram_pico: tuple[float, float], identacao: int = 0) -> None:
    identacao_string = ' ' * identacao * 4

    diferenca_ram = round(ram[1] - ram[0], 2)
    diferenca_vram = round(vram[1] - vram[0], 2)
    diferenca_vram_pico = round(vram_pico[1] - vram_pico[0], 2)

    if diferenca_ram:
        print(f'{identacao_string}• {ram[0]:.2f}MB RAM inicial e {ram[1]:.2f}MB final ({diferenca_ram:+}MB RAM)')

    if diferenca_vram:
        print(f'{identacao_string}• {vram[0]:.2f}MB VRAM inicial e {vram[1]:.2f}MB final ({diferenca_vram:+}MB VRAM)')

    if diferenca_vram_pico:
        print(f'{identacao_string}• {vram_pico[0]:.2f}MB pico de VRAM inicial e {vram_pico[1]:.2f}MB final ({diferenca_vram_pico:+}MB VRAM)')

    print()


def _formata_segundos(segundos: int) -> str:
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos_restantes = segundos % 60

    # se o tempo decorrido é maior que 60 segundos, remove a casa decimal
    if horas or minutos:
        segundos_restantes = round(segundos_restantes)

    formatado = []

    if horas > 0:
        formatado.append(f'{horas:.0f}h')

    if minutos > 0:
        formatado.append(f'{minutos:.0f}m')

    if segundos_restantes > 0:
        formatado.append(f'{segundos_restantes:.2f}s')

    return ' '.join(formatado)


def _obtem_ram() -> int:
    from os import getpid
    from psutil import Process

    pid = getpid()
    processo = Process(pid)

    ram_atual_mb = processo.memory_info().rss / 1024**2

    return ram_atual_mb

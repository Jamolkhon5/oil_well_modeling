"""
Модуль с настройками, путями и константами проекта.
"""

import os
from pathlib import Path

# Пути к директориям
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Пути к файлам данных
PZAB_FILE = os.path.join(DATA_DIR, 'pzab_quarterly.csv')
PPL_FILE = os.path.join(DATA_DIR, 'ppl_quarterly.xlsx')
ARCGIS_FILE = os.path.join(DATA_DIR, 'arcgis_fund.csv')
GDI_VNR_FILE = os.path.join(DATA_DIR, 'gdi_vnr.xlsx')
GDI_REINT_FILE = os.path.join(DATA_DIR, 'gdi_reinterpretation.xlsx')
GTP_FILE = os.path.join(DATA_DIR, 'gtp.xlsx')
NNT_NGT_FILE = os.path.join(DATA_DIR, 'nnt_NGT.xlsx')

# Кодировки файлов
PZAB_ENCODING = 'utf-8'
ARCGIS_ENCODING = 'utf-8'

# Параметры для модуля 1: Подбор относительных фазовых проницаемостей
OFP_PARAMS = {
    'Swo': 0.42,        # Остаточная водонасыщенность
    'Swk': 0.677,       # Водонасыщенность при остаточной нефтенасыщенности
    'krwk': 0.135,      # Конечное значение относительной водопронецаемости
    'krok': 1.0,        # Конечное значение относительной нефтепронецаемости
    'mu_w': 0.45,       # Динамическая вязкость воды, сПз
    'mu_o': 1.3,        # Динамическая вязкость нефти, сПз
    'krw_min': 1.0,     # Подгонная степень 1 для функции Кори ОФП по воде (минимум)
    'krw_max': 1.5,     # Подгонная степень 1 для функции Кори ОФП по воде (максимум)
    'kro_min': 0.1,     # Подгонная степень 2 для функции Кори ОФП по нефти (минимум)
    'kro_max': 9.0,     # Подгонная степень 2 для функции Кори ОФП по нефти (максимум)
    'kn1': 1.0,         # Подгонный коэффициент по проницаемости
    'cw': 4.27e-5,      # Сжимаемость воды, 1/атм
    'co': 12e-5,        # Сжимаемость нефти, 1/атм
    'cf': 3.3e-5,       # Сжимаемость пласта, 1/атм
}

# Диапазоны параметров (минимум/максимум)
OFP_RANGES = {
    'Swo': (0.025, 0.6),            # Остаточная водонасыщенность
    'Swk': (0.54, 0.9),             # Водонасыщенность при остаточной нефтенасыщенности
    'krwk': (0.06, 0.88),           # Конечное значение относительной водопронецаемости
    'krok': (1.0, 1.0),             # Конечное значение относительной нефтепронецаемости
    'mu_w': (0.24, 19.0),           # Динамическая вязкость воды, сПз
    'mu_o': (0.013, 111.0),         # Динамическая вязкость нефти, сПз
    'krw_min': (0.55, 4.0),         # Подгонная степень 1 для функции Кори ОФП по воде (минимум)
    'krw_max': (-0.5, 3.5),         # Подгонная степень 1 для функции Кори ОФП по воде (максимум)
    'kro_min': (0.1, 9.0),          # Подгонная степень 2 для функции Кори ОФП по нефти (минимум)
    'kro_max': (2.5, 30.0),         # Подгонная степень 2 для функции Кори ОФП по нефти (максимум)
    'cw': (0.85e-5, 44.7e-5),       # Сжимаемость воды, 1/атм
    'co': (2e-5, 75e-5),            # Сжимаемость нефти, 1/атм
    'cf': (0.193e-5, 10.2e-5),      # Сжимаемость пласта, 1/атм
}

# Прочие константы
TOLERANCE = 0.001  # Допустимая погрешность для модуля 2
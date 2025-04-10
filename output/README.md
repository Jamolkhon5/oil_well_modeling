# Выходные данные расчетной схемы Пушкиной Т.В.

В этой директории хранятся все выходные данные модели и результаты расчетов.

## Структура директории

- `results/` - Основные результаты расчетов в различных форматах
  - `phase_permeability/` - Результаты модуля 1 (Подбор относительных фазовых проницаемостей)
  - `regression_model/` - Результаты модуля 2 (Подбор итеративной регрессионной моделью)
  - `pressure_calculation/` - Результаты модуля 3 (Расчет Рпл в нефтяных скважинах)
  - `pressure_recovery/` - Результаты модуля 4 (Подбор времени восстановления давления)
  - `skin_curve/` - Результаты модуля 5 (Подбор кривой увеличения SKIN)
  - `filter_reduction/` - Результаты модуля 6 (Подбор к-та уменьшения работающей части фильтра)
  - `fracture_length/` - Результаты модуля 7 (Подбор коэффициентов для расчета полудлин трещин)
  - `production_wells/` - Результаты модуля 8 (Расчет добывающих скважин)
  - `model_parameters.xlsx` - Сводная таблица параметров всех моделей

- `figures/` - Графики и визуализации результатов
  - `phase_permeability_curves.png` - График относительных фазовых проницаемостей
  - `pressure_changes.png` - График изменения пластового давления
  - `recovery_times.png` - График времени восстановления давления
  - `skin_curve.png` - График изменения скин-фактора
  - `filter_reduction_curve.png` - График уменьшения работающей части фильтра
  - `fracture_length_curve.png` - График зависимости полудлины трещины от объема закачки
  - `production_profiles.png` - График профилей добычи
  - `cumulative_production.png` - График накопленной добычи
  - `summary_dashboard.png` - Сводная панель со всеми графиками

- `reports/` - Отчеты о результатах расчетов
  - `report_YYYYMMDD_HHMMSS.pdf` - Автоматически сгенерированный отчет с датой и временем создания
  - `summary_report.pdf` - Обобщенный отчет о результатах

## Форматы данных

Результаты доступны в нескольких форматах:
- CSV (для импорта в другие программы)
- Excel (для удобного просмотра и анализа)
- JSON (для программного доступа)

## Использование результатов

1. **Визуальный анализ**:
   - Просмотрите графики в директории `figures/` для визуального анализа результатов
   - Изучите отчеты в директории `reports/` для получения полного представления о расчетах

2. **Дальнейший анализ данных**:
   - Используйте файлы CSV или Excel из директории `results/` для импорта в другие программы анализа
   - Применяйте JSON-файлы для программного доступа к результатам

3. **Получение параметров моделей**:
   - Файл `model_parameters.xlsx` содержит все калиброванные параметры моделей

## Примечания

- В директории сохраняются только последние результаты расчетов
- При каждом новом запуске расчетов старые результаты перезаписываются
- Для сохранения истории расчетов рекомендуется копировать важные результаты в отдельную директорию
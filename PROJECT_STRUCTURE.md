# Project Structure

Краткая карта проекта

## Основные файлы

- `README.md` — исходное условие
- `sell.csv` — данные продаж для 60 
- `notebooks/lab3_timeseries.ipynb` — основной ноутбук базовой части: загрузка данных, три фильтра, графики, корреляции, автокорреляция и выводы.
- `notebooks/lab3_hard_wavelet_sensors.ipynb` — дополнительный ноутбук hard-части: ручная Haar wavelet-фильтрация и анализ Kaggle-датасета с акселерометром и гироскопом.

## Код

- `src/data_loader.py` — загрузка `sell.csv`, расчет варианта по ИСУ, извлечение нужного блока данных.
- `src/analysis.py` — функции анализа временных рядов: корреляции, автокорреляция, доминирующий лаг, описательная статистика.
- `src/metrics.py` — метрики сравнения фильтрации: RMSE и сглаженность.
- `src/hard_data_loader.py` — загрузка и подготовка Kaggle-датасета для hard-части.

## Фильтры

- `src/filters/moving_average.py` — скользящее среднее.
- `src/filters/kalman.py` — одномерный фильтр Калмана с моделью случайного блуждания.
- `src/filters/savitzky_golay.py` — ручная реализация фильтра Савицкого-Голея.
- `src/filters/wavelet.py` — ручная Haar wavelet-фильтрация для hard-части.

## Данные hard-части

Hard-ноутбук использует Kaggle dataset:

`krishujeniya/fitness-tracker-accelerometer-and-gyroscope-data`

При запуске данные скачиваются в `data/hard/`. 


## Как запускать

1. Открыть проект в корне репозитория.
2. Для базовой части запустить `notebooks/lab3_timeseries.ipynb`.
3. Для hard-части запустить `notebooks/lab3_hard_wavelet_sensors.ipynb`.

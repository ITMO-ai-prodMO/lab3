"""Сборщик research.ipynb. Запуск: ``python build_notebook.py`` из папки ``lab_03/``."""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "research.ipynb"
CSV = "sell.csv"
ISU_LIST = [465430, 467715]


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


cells: list[nbf.NotebookNode] = []

# ============================================================================
# Заголовок (центрированный, как в lab1)
# ============================================================================
cells += [
    md(
        '<div style="text-align:center">\n'
        "\n"
        "# Лабораторная работа №3\n"
        "## Анализ временных рядов\n"
        "### Вариант 25\n"
        "\n"
        "</div>"
    ),
]

# ============================================================================
# Постановка задачи
# ============================================================================
cells += [
    md(
        "## Постановка задачи\n"
        "\n"
        "Дан набор данных `sell.csv` - продажи **60 вариантов** товара за "
        "**$N = 50$ дней**. Для каждого варианта пять натуральных метрик "
        "($\\text{мыло, порошок, средство, краска, пена}$) в штуках и "
        "целевая переменная $\\text{прибыль}$ в тыс. руб.\n"
        "\n"
        "Номер варианта определяется как\n"
        "\n"
        "$$V = \\bigl(\\, \\mathrm{ISU}_1 + \\mathrm{ISU}_2 \\,\\bigr) \\bmod 60 "
        "\\;=\\; (465430 + 467715) \\bmod 60 \\;=\\; 25.$$\n"
        "\n"
        "Лабораторная состоит из четырёх задач:\n"
        "\n"
        "- **Задача 1.** Реализовать три фильтра + hard-добавку (вейвлет/LMS).\n"
        "- **Задача 2.** Прогнать фильтры по данным варианта, построить графики.\n"
        "- **Задача 3.** Вычислить характеристики ряда: тренд, стационарность, "
        "периодичность, корреляции.\n"
        "- **Задача 4 (Hard).** Найти данные с двумя датчиками и применить "
        "фильтрацию + анализ.\n"
        "\n"
        "Вся бизнес-логика - в [`./src/`](./src/); ноутбук импортирует и "
        "визуализирует."
    ),
]

# ============================================================================
# Подготовка окружения
# ============================================================================
cells += [
    md("## Подготовка окружения"),
    code(
        "# Поиск lab_03/src/ и фиксация рабочей директории\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "\n"
        "_cwd = Path(os.getcwd()).resolve()\n"
        "for _candidate in [_cwd, *_cwd.parents]:\n"
        "    _src = _candidate / 'lab_03' / 'src'\n"
        "    if _src.is_dir() and (_src / 'utils.py').exists():\n"
        "        break\n"
        "    _src = _candidate / 'src'\n"
        "    if _src.is_dir() and (_src / 'utils.py').exists():\n"
        "        break\n"
        "else:\n"
        "    raise RuntimeError('Не нашёл lab_03/src/ - запусти ноутбук из lab_03/ или её родителя')\n"
        "os.chdir(_src.parent)\n"
        "sys.path.insert(0, str(_src))\n"
        "print('Working dir:', os.getcwd())"
    ),
    code(
        "# Импорты и стиль\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "import utils, eda, periodicity, correlations, filters, wavelet_lms, task4_sensors\n"
        "\n"
        "utils.seed_everything(42)\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
        "pd.set_option('display.precision', 4)\n"
        "plt.rcParams['figure.dpi'] = 110"
    ),
    code(
        "# Загрузка варианта 25\n"
        f"ISU = {ISU_LIST!r}\n"
        "V = utils.compute_variant(ISU)\n"
        f"df = utils.load_variant('{CSV}', V)\n"
        "print(f'Вариант: {V},  shape: {df.shape},  колонки: {list(df.columns)}')\n"
        "df.head()"
    ),
]

# ============================================================================
# Задача 1. Фильтры
# ============================================================================
cells += [
    md(
        "## Задача 1. Реализация фильтров\n"
        "\n"
        "**1.1 Скользящее среднее и экспоненциальное сглаживание.** "
        "Центрированное MA с симметричным `edge`-паддингом и Brown\n"
        "\n"
        "$$\\hat x_t = \\alpha\\, x_t + (1 - \\alpha)\\, \\hat x_{t-1}, "
        "\\qquad \\hat x_0 = x_0,\\quad 0 < \\alpha \\le 1.$$\n"
        "\n"
        "**1.2 Калман на модели случайного блуждания (RW).** Скалярный, с "
        "обратным RTS-проходом:\n"
        "\n"
        "$$x_t = x_{t-1} + w_t,\\; w_t \\sim \\mathcal{N}(0, Q), \\qquad "
        "z_t = x_t + v_t,\\; v_t \\sim \\mathcal{N}(0, R).$$\n"
        "\n"
        "Параметры $Q, R$ оцениваются методом моментов из вторых разностей "
        "наблюдений.\n"
        "\n"
        "**1.3 Фильтр Савицкого-Голея.** Локальная полиномиальная регрессия "
        "степени $p$ на окне $w$ через `scipy.signal.savgol_filter`.\n"
        "\n"
        "**1.4 Hard.** *DWT-денойзинг*: разложение `pywt.wavedec`, оценка "
        "$\\sigma$ через MAD/0.6745 по первому уровню деталей, универсальный "
        "порог Donoho $\\lambda = \\sigma\\sqrt{2\\ln N}$ + soft-thresholding. "
        "*LMS-адаптивный* в режиме предсказания по $n_\\text{taps}$ "
        "предыдущим значениям, шаг $\\mu$ - из условия устойчивости."
    ),
    md(
        "### Демонстрация на синтетике\n"
        "\n"
        "Тестовый сигнал длиной 200 точек:\n"
        "\n"
        "$$x_t = \\sin\\!\\bigl(2\\pi t / 30\\bigr) + "
        "\\tfrac{1}{2}\\sin\\!\\bigl(2\\pi t / 8\\bigr) + \\varepsilon_t, "
        "\\qquad \\varepsilon_t \\sim \\mathcal{N}(0,\\, 0.5^2).$$\n"
        "\n"
        "Истина известна - считаем честный MSE."
    ),
    code(
        "# Синтетический сигнал и все 6 фильтров\n"
        "rng = np.random.default_rng(42)\n"
        "n = 200\n"
        "t = np.arange(n)\n"
        "true = np.sin(2*np.pi*t/30) + 0.5*np.sin(2*np.pi*t/8)\n"
        "noisy = true + rng.normal(0, 0.5, n)\n"
        "demo = pd.DataFrame({\n"
        "    'true': true, 'noisy': noisy,\n"
        "    'MA(7)':        filters.moving_average(noisy, window=7),\n"
        "    'Exp(0.3)':     filters.exponential_smoothing(noisy, alpha=0.3),\n"
        "    'Kalman_RW':    filters.kalman_1d_rw(noisy),\n"
        "    'SavGol(11,2)': filters.savitzky_golay(noisy, window=11, polyorder=2),\n"
        "    'DWT(db4)':     wavelet_lms.dwt_denoise(noisy)['denoised'][:n],\n"
        "    'LMS(5)':       wavelet_lms.lms_denoise(noisy, n_taps=5)['denoised'],\n"
        "})\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.plot(t, demo['noisy'], color='lightgray', lw=1, label='noisy')\n"
        "ax.plot(t, demo['true'],  color='black',     lw=2, label='true')\n"
        "for col, c in zip(['MA(7)','Exp(0.3)','Kalman_RW','SavGol(11,2)','DWT(db4)','LMS(5)'],\n"
        "                   ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']):\n"
        "    ax.plot(t, demo[col], lw=1.1, label=col, color=c, alpha=0.85)\n"
        "ax.set_title('Фильтры на синтетике: sin + sin + N(0, 0.5^2)')\n"
        "ax.legend(loc='upper right', ncol=2, fontsize=9)\n"
        "plt.tight_layout(); plt.savefig('figures/filters_demo_synthetic.png', dpi=120)\n"
        "plt.show()"
    ),
    code(
        "# MSE и smoothing_degree относительно истины\n"
        "demo_metrics = pd.DataFrame({\n"
        "    col: filters.filter_metrics(true, demo[col].values)\n"
        "    for col in ['MA(7)','Exp(0.3)','Kalman_RW','SavGol(11,2)','DWT(db4)','LMS(5)']\n"
        "}).T\n"
        "demo_metrics.round(4)"
    ),
]

# ============================================================================
# Задача 2. Применение к варианту 25
# ============================================================================
cells += [
    md(
        "## Задача 2. Применение фильтров к данным варианта 25\n"
        "\n"
        "По требованию README - графики обязательны. Рассматриваем две "
        "ключевые метрики: $\\text{прибыль}$ и $\\text{мыло}$ (самая объёмная "
        "номенклатура и, как покажет Задача 3, главный драйвер прибыли).\n"
        "\n"
        "Без истинного сигнала MSE отражает не точность, а интенсивность "
        "сглаживания. Содержательная метрика -\n"
        "\n"
        "$$\\text{smoothing\\_degree}(\\hat x) = 1 - \\frac{\\mathrm{Var}(\\hat x)}{\\mathrm{Var}(x)}.$$"
    ),
    code(
        "# Прогон всех фильтров по прибыли и мылу варианта 25\n"
        "series_profit = df[utils.PROFIT_COL]\n"
        "series_soap   = df['мыло']\n"
        "\n"
        "core_profit = filters.apply_all_core(series_profit)\n"
        "hard_profit = wavelet_lms.apply_all_hard(series_profit)\n"
        "core_soap   = filters.apply_all_core(series_soap)\n"
        "hard_soap   = wavelet_lms.apply_all_hard(series_soap)\n"
        "core_profit.head()"
    ),
    code(
        "# Таблица метрик качества фильтрации для прибыли\n"
        "all_filtered_profit = pd.concat([core_profit, hard_profit.drop(columns='original')], axis=1)\n"
        "rows = {}\n"
        "for col in all_filtered_profit.columns:\n"
        "    if col == 'original':\n"
        "        continue\n"
        "    m = filters.filter_metrics(series_profit.values, all_filtered_profit[col].values)\n"
        "    rows[col] = {k.upper() if k.upper() in ('MSE','MAE','RMSE') else k: v for k, v in m.items()}\n"
        "metrics_profit = pd.DataFrame(rows).T\n"
        "metrics_profit.round(4)"
    ),
    code(
        "# График: базовые и хард-фильтры на прибыли\n"
        "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
        "axes[0].plot(series_profit.index, series_profit.values, color='lightgray', label='original')\n"
        "for col, c in zip(['MA(7)','Exp(a=0.5)','Kalman_RW','SavGol(7,2)'],\n"
        "                   ['#1f77b4','#ff7f0e','#2ca02c','#d62728']):\n"
        "    axes[0].plot(series_profit.index, core_profit[col].values, lw=1.4, label=col, color=c)\n"
        "axes[0].set_title('Базовые фильтры (Задачи 1.1-1.3): прибыль варианта 25')\n"
        "axes[0].set_ylabel('тыс. руб.'); axes[0].legend(ncol=3, fontsize=9)\n"
        "\n"
        "axes[1].plot(series_profit.index, series_profit.values, color='lightgray', label='original')\n"
        "for col, c in zip(['DWT(db4)','DWT(sym4)','LMS(5,auto)'],\n"
        "                   ['#9467bd','#8c564b','#e377c2']):\n"
        "    axes[1].plot(series_profit.index, hard_profit[col].values, lw=1.4, label=col, color=c)\n"
        "axes[1].set_title('Хард-фильтры (Задача 1.4): прибыль варианта 25')\n"
        "axes[1].set_xlabel('день'); axes[1].set_ylabel('тыс. руб.'); axes[1].legend(ncol=3, fontsize=9)\n"
        "plt.tight_layout(); plt.savefig('figures/task2_filters_profit.png', dpi=120)\n"
        "plt.show()"
    ),
    code(
        "# Те же фильтры на мыле\n"
        "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
        "axes[0].plot(series_soap.index, series_soap.values, color='lightgray', label='original')\n"
        "for col, c in zip(['MA(7)','Exp(a=0.5)','Kalman_RW','SavGol(7,2)'],\n"
        "                   ['#1f77b4','#ff7f0e','#2ca02c','#d62728']):\n"
        "    axes[0].plot(series_soap.index, core_soap[col].values, lw=1.4, label=col, color=c)\n"
        "axes[0].set_title('Базовые фильтры: мыло, вариант 25')\n"
        "axes[0].set_ylabel('шт.'); axes[0].legend(ncol=3, fontsize=9)\n"
        "\n"
        "axes[1].plot(series_soap.index, series_soap.values, color='lightgray', label='original')\n"
        "for col, c in zip(['DWT(db4)','DWT(sym4)','LMS(5,auto)'],\n"
        "                   ['#9467bd','#8c564b','#e377c2']):\n"
        "    axes[1].plot(series_soap.index, hard_soap[col].values, lw=1.4, label=col, color=c)\n"
        "axes[1].set_title('Хард-фильтры: мыло, вариант 25')\n"
        "axes[1].set_xlabel('день'); axes[1].set_ylabel('шт.'); axes[1].legend(ncol=3, fontsize=9)\n"
        "plt.tight_layout(); plt.savefig('figures/task2_filters_soap.png', dpi=120)\n"
        "plt.show()"
    ),
    code(
        "# Калман RTS-сглаживатель с зоной неопределенности +-2*ско\n"
        "kf = filters.KalmanFilter1D(Q=0.6, R=2.6)\n"
        "x_s, P_s = kf.smooth(series_profit.values)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.plot(series_profit.index, series_profit.values, color='lightgray', label='наблюдения')\n"
        "ax.plot(series_profit.index, x_s, color='#2ca02c', lw=1.8, label='Kalman RTS smoother')\n"
        "ax.fill_between(series_profit.index, x_s - 2*np.sqrt(P_s), x_s + 2*np.sqrt(P_s),\n"
        "                color='#2ca02c', alpha=0.2, label='+-2*ско')\n"
        "ax.set_title('Калман 1D (RW): прибыль варианта 25 с зоной неопределённости')\n"
        "ax.set_xlabel('день'); ax.set_ylabel('тыс. руб.'); ax.legend()\n"
        "plt.tight_layout(); plt.savefig('figures/task2_kalman_uncertainty.png', dpi=120)\n"
        "plt.show()"
    ),
]

# ============================================================================
# Задача 3. Анализ
# ============================================================================
cells += [
    md(
        "## Задача 3. Характеристики временного ряда\n"
        "\n"
        "Считаем: описательные статистики, линейный и Манна-Кендалла тренд, "
        "ADF/KPSS, ACF/PACF, FFT после удаления тренда, STL-декомпозицию, "
        "Pearson/Spearman/частные корреляции, кросс-корреляции с лагами, "
        "взаимную информацию.\n"
        "\n"
        "**Ограничение $N=50$.** ACF теряет мощность на лагах $> N/2 - 1 = 24$; "
        "KPSS p-value табулирован только в $[0.01, 0.10]$ - верхняя граница "
        "клампится; STL требует $\\ge 4$ полных циклов, т.е. $P \\le N/3 \\approx 16$. "
        "Все спектральные оценки за пределами - ориентировочные."
    ),
    md("### 3.1 Описательная статистика, тренд, стационарность"),
    code(
        "# Описательная статистика\n"
        "stats = eda.descriptive_stats(df)\n"
        "stats.round(3)"
    ),
    code(
        "# Тренд: линейная регрессия (lr) и Манна-Кендалла (mk)\n"
        "trend = {col: eda.trend_analysis(df[col], name=col) for col in utils.ALL_METRIC_COLS}\n"
        "pd.DataFrame(trend).T[['lr_slope','lr_p','mk_tau','mk_p']].round(4)"
    ),
    code(
        "# Стационарность: тесты ADF и KPSS\n"
        "stat = {col: eda.stationarity_tests(df[col]) for col in utils.ALL_METRIC_COLS}\n"
        "pd.DataFrame({c: {'adf_p': v['adf_p'], 'kpss_p': v['kpss_p'], 'verdict': v['verdict']}\n"
        "              for c, v in stat.items()}).T"
    ),
    code(
        "# Обзор шести рядов: сырая + скользящее среднее с полосой +-2*ско\n"
        "eda.plot_overview(df, save_path='figures/eda_overview.png')\n"
        "plt.show()"
    ),
    code(
        "# Распределения: гистограмма и оценка плотности\n"
        "eda.plot_distributions(df, save_path='figures/eda_distributions.png')\n"
        "plt.show()"
    ),
    md(
        "### 3.2 Периодичность\n"
        "\n"
        "Силу сезонности оцениваем как\n"
        "\n"
        "$$F_s = \\max\\!\\left(0,\\; 1 - \\frac{\\mathrm{Var}(R)}{\\mathrm{Var}(S + R)}\\right),$$\n"
        "\n"
        "где $S, R$ - сезонная и остаточная компоненты STL-разложения."
    ),
    code(
        "# ACF и PACF: значимые лаги для прибыли\n"
        "acf_profit = periodicity.acf_pacf(df[utils.PROFIT_COL], nlags=20)\n"
        "print('Значимые лаги ACF  (прибыль):', acf_profit['significant_acf'])\n"
        "print('Значимые лаги PACF (прибыль):', acf_profit['significant_pacf'])"
    ),
    code(
        "periodicity.plot_acf_pacf(df[utils.PROFIT_COL],\n"
        "                           save_path='figures/acf_pacf_прибыль.png', nlags=20)\n"
        "plt.show()"
    ),
    code(
        "# FFT-спектр после удаления тренда: топ-3 доминирующих периода\n"
        "fft_profit = periodicity.fft_spectrum(df[utils.PROFIT_COL], detrend=True)\n"
        "for d in fft_profit['dominant_periods']:\n"
        "    print(f\"period = {d['period_days']:5.2f} д.    freq = {d['freq']:.3f}    power = {d['power']:.2f}\")"
    ),
    code(
        "periodicity.plot_fft(df[utils.PROFIT_COL],\n"
        "                      save_path='figures/fft_прибыль.png', top_k=3)\n"
        "plt.show()"
    ),
    code(
        "# STL-декомпозиция: гипотезы периода P=7 (недельная) и P=12\n"
        "stl7  = periodicity.stl_decomposition(df[utils.PROFIT_COL], period=7)\n"
        "stl12 = periodicity.stl_decomposition(df[utils.PROFIT_COL], period=12)\n"
        "pd.DataFrame({'P=7':  {'Fs_stl': stl7['Fs_stl'],  'Ft_stl': stl7['Ft_stl']},\n"
        "              'P=12': {'Fs_stl': stl12['Fs_stl'], 'Ft_stl': stl12['Ft_stl']}}).round(3)"
    ),
    code(
        "periodicity.plot_decomposition(df[utils.PROFIT_COL], period=7,\n"
        "                                save_path='figures/stl_прибыль_p7.png')\n"
        "plt.show()"
    ),
    code(
        "# Поиск надёжного периода в диапазоне P <= N/3\n"
        "search = periodicity.find_dominant_period(df[utils.PROFIT_COL])\n"
        "print(f'Global best       : P = {search[\"best_period\"]}, Fs = {search[\"best_Fs\"]:.3f}')\n"
        "print(f'Надёжный (P<=N/3) : P = {search[\"best_reliable_period\"]}, Fs = {search[\"best_reliable_Fs\"]:.3f}')"
    ),
    md(
        "### 3.3 Корреляции и зависимости\n"
        "\n"
        "Pearson + Spearman + частные корреляции через инверсию матрицы "
        "Pearson: $\\mathbf{P} = -\\mathrm{cov}(X)^{-1}$, нормировка "
        "$p_{ij} = -\\mathbf{P}_{ij}\\big/\\sqrt{\\mathbf{P}_{ii}\\mathbf{P}_{jj}}$. "
        "Дополнительно - кросс-корреляции с лагами от $-15$ до $+15$ и "
        "матрица взаимной информации для проверки нелинейных связей."
    ),
    code(
        "# Pearson: матрица корреляций и матрица p-value\n"
        "corr_p, pval = correlations.correlation_with_pvalues(df, method='pearson')\n"
        "print('Pearson r:'); display(corr_p.round(3))\n"
        "print('p-value:');  display(pval.round(4))"
    ),
    code(
        "# Spearman (ранговая) для проверки монотонных связей\n"
        "correlations.correlation_matrix(df, method='spearman').round(3)"
    ),
    code(
        "# Частные корреляции через инверсию матрицы Pearson\n"
        "part = correlations.partial_correlation_matrix(df); part.round(3)"
    ),
    code(
        "# Heatmap: Pearson и частные корреляции рядом\n"
        "correlations.plot_corr_heatmap(corr_p, save_path='figures/corr_pearson.png', title='Pearson')\n"
        "plt.show()\n"
        "correlations.plot_corr_heatmap(part,   save_path='figures/corr_partial.png', title='Partial')\n"
        "plt.show()"
    ),
    code(
        "# Lead-lag таблица: лучший лаг для каждой пары метрик\n"
        "ll = correlations.lead_lag_table(df, max_lag=10); ll.head(10)"
    ),
    code(
        "# Кросс-корреляция прибыль и мыло с лагами\n"
        "correlations.plot_xcorr(df[utils.PROFIT_COL], df['мыло'], max_lag=15,\n"
        "                         save_path='figures/xcorr_прибыль_vs_мыло.png',\n"
        "                         title='Прибыль - Мыло (лаги)')\n"
        "plt.show()"
    ),
    code(
        "# Mutual information: нелинейные зависимости\n"
        "mi = correlations.mutual_information_matrix(df, bins=8); mi.round(3)"
    ),
    code(
        "# Близкие варианты по паттерну прибыли (топ-10)\n"
        f"all_v = utils.load_all_variants('{CSV}')\n"
        "sim = correlations.cross_variant_profit_corr(all_v, target_variant=V)\n"
        "sim.head(10)"
    ),
]

# ============================================================================
# Задача 4. Двойные датчики (Hard)
# ============================================================================
cells += [
    md(
        "## Задача 4 (Hard). Двойные датчики\n"
        "\n"
        "Источник: **UCI Air Quality Dataset** "
        "([archive.ics.uci.edu/.../AirQualityUCI.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip)) - "
        "почасовые измерения CO в итальянском городе, 2004-2005 гг. "
        "Два датчика:\n"
        "\n"
        "- $z_1 = \\text{CO(GT)}$ - электрохимический эталон, мг/м^3;\n"
        "- $z_2 = \\text{PT08.S1(CO)}$ - оксид-металлический, сырые ADC-отсчёты "
        "(перед фьюжн аффинно выровнен по МНК).\n"
        "\n"
        "**Калман с последовательным апдейтом двумя измерениями.** На каждом "
        "шаге:\n"
        "\n"
        "$$\\hat x_t^- = \\hat x_{t-1}, \\qquad P_t^- = P_{t-1} + Q,$$\n"
        "$$K_1 = \\frac{P_t^-}{P_t^- + R_1}, \\quad "
        "\\hat x_t' = \\hat x_t^- + K_1 (z_{1,t} - \\hat x_t^-), \\quad "
        "P_t' = (1 - K_1) P_t^-,$$\n"
        "$$K_2 = \\frac{P_t'}{P_t' + R_2}, \\quad "
        "\\hat x_t = \\hat x_t' + K_2 (z_{2,t} - \\hat x_t'), \\quad "
        "P_t = (1 - K_2) P_t'.$$\n"
        "\n"
        "Если сеть недоступна, модуль автоматически переключается на "
        "синтетический dual-sensor с прозрачной физической моделью."
    ),
    code(
        "# Загрузка UCI Air Quality (fallback на синтетику если нет сети)\n"
        "ds, meta = task4_sensors.load_dual_sensor()\n"
        "print('Источник :', meta.get('source'))\n"
        "print('Описание :', meta.get('description'))\n"
        "print('Размер   :', len(ds))\n"
        "ds.head()"
    ),
    code(
        "# Калман-фьюжн двух датчиков с автоматической оценкой Q, R1, R2\n"
        "fused = task4_sensors.fuse_sensors_kalman(ds['sensor_1'].values, ds['sensor_2'].values)\n"
        "print(f'Параметры:  Q = {fused[\"Q\"]:.4f},  R1 = {fused[\"R1\"]:.4f},  R2 = {fused[\"R2\"]:.4f}')\n"
        "print(f'Дисперсии:  z1 = {ds[\"sensor_1\"].var():.3f},  z2 = {ds[\"sensor_2\"].var():.3f},  fused = {fused[\"fused\"].var():.3f}')"
    ),
    code(
        "# Анализ остатков фьюжн и каждого датчика\n"
        "res = task4_sensors.analyze_residuals(fused['fused'], ds['sensor_1'].values, ds['sensor_2'].values)\n"
        "pd.Series(res).round(4)"
    ),
    code(
        "# Детектор аномалий по 3*ско от скользящего среднего\n"
        "anom = task4_sensors.detect_anomalies(ds, fused['fused'])\n"
        "print(f'Найдено аномалий: {len(anom)}')\n"
        "anom.head()"
    ),
    code(
        "# Регенерация и отображение всех графиков задачи 4\n"
        "from IPython.display import Image, display\n"
        "task4_sensors.plot_raw_sensors(ds, meta)\n"
        "task4_sensors.plot_fused_kalman(ds, fused)\n"
        "task4_sensors.plot_residuals(ds, fused['fused'])\n"
        "task4_sensors.plot_anomalies(ds, fused['fused'], anom)\n"
        "task4_sensors.plot_snr_improvement(res)\n"
        "for p in ['task4_raw_sensors.png','task4_fused_kalman.png',\n"
        "         'task4_residuals_per_sensor.png','task4_anomalies.png',\n"
        "         'task4_snr_improvement.png']:\n"
        "    display(Image(f'figures/{p}'))"
    ),
]

# ============================================================================
# Интерпретация результатов
# ============================================================================
cells += [
    md(
        "## Интерпретация результатов\n"
        "\n"
        "**Фильтры.** Сводные метрики прибыли (вариант 25):\n"
        "\n"
        "| Фильтр | MSE | smoothing_degree |\n"
        "|---|---|---|\n"
        "| MA(7) | $2.42$ | $0.81$ |\n"
        "| Exp(a = 0.5) | $1.03$ | $0.55$ |\n"
        "| Holt | $6.24$ | $-0.07$ |\n"
        "| **Kalman_RW** | $\\mathbf{1.93}$ | $\\mathbf{0.82}$ |\n"
        "| SavGol(7,2) | $2.05$ | $0.55$ |\n"
        "| DWT(sym4) | $2.08$ | $0.75$ |\n"
        "| LMS(5) | $3.32$ | $0.59$ |\n"
        "\n"
        "Калман даёт лучший компромисс «удалить шум, не уйти от наблюдений». "
        "Отрицательный smoothing_degree у Холта - диагностика стационарности "
        "ряда: трендовая компонента вырождается в шум.\n"
        "\n"
        "**Ряд варианта 25.** Стационарный шум вокруг постоянного среднего:\n"
        "\n"
        "- ни линейный, ни Манна-Кендалла тренд не значимы ($p > 0.37$ во всех 6 рядах);\n"
        "- ADF + KPSS дают вердикт *stationary* для всех метрик;\n"
        "- ACF без значимых лагов на 95% CI; FFT-пик у частоты Найквиста - артефакт;\n"
        "- надёжная сезонная гипотеза - $P \\approx 8$ дней, $F_s \\approx 0.32$ (слабая).\n"
        "\n"
        "**Корреляции - главный качественный вывод.**\n"
        "\n"
        "| Пара | Pearson $r$ | Partial $r$ | p-value |\n"
        "|---|---|---|---|\n"
        "| мыло - прибыль | $\\mathbf{0.73}$ | $\\mathbf{0.72}$ | $2.2 \\cdot 10^{-9}$ |\n"
        "| мыло - средство | $0.71$ | $0.69$ | $1.1 \\cdot 10^{-8}$ |\n"
        "| **средство - прибыль** | $0.51$ | $\\mathbf{-0.03}$ | $1.5 \\cdot 10^{-4}$ |\n"
        "\n"
        "Pearson-связь средства с прибылью **мнимая** - обнуляется при контроле "
        "остальных метрик. Реальный драйвер прибыли - мыло, синхронно (best "
        "lag = 0). MI не открывает нелинейных связей сверх линейных. Среди "
        "60 вариантов вариант 25 не имеет близких аналогов "
        "($\\max\\,r \\approx 0.36$).\n"
        "\n"
        "**UCI dual-sensor.** Kalman-фьюжн снижает дисперсию остатков на "
        "**63.5%** относительно эталонного датчика и на **49.6%** относительно "
        "оксид-металлического. Pearson сырых датчиков $0.898$ - высокая "
        "согласованность. Кросс-корреляция остатков $0.46$ - линейный фьюжн "
        "не предельный, остаётся общий компонент атмосферного шума, который "
        "сняла бы только нелинейная модель.\n"
        "\n"
        "**Воспроизводимость.**\n"
        "\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "jupyter nbconvert --to notebook --execute research.ipynb --output research.ipynb\n"
        "```\n"
        "\n"
        "Seed зафиксирован, внешние данные кешируются автоматически."
    ),
]

# ============================================================================
# Write notebook
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", NB_PATH, "with", len(cells), "cells")

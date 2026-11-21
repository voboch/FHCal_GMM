# Отчёты FHCal

[Все отчёты одним PDF с закладками](FHCal_Reports_Current.pdf).

| Эксперимент | Материалы |
| --- | --- |
| Мюонный пилот аппроксимации отклика | [PDF](FHCal_Gaussian_Mixture_Pilot.pdf) · [Markdown и рисунки](../src/gaussian_projection/RESULTS.md) |
| Протонный пилот, число компонент и занятость | [PDF](FHCal_Proton_Projection_Pilot.pdf) · [Markdown и рисунки](../src/gaussian_projection/proton_pilot/RESULTS.md) |
| Условный банк и стохастический декодер | [PDF](FHCal_Stochastic_Decoder_Report.pdf) · [анализ и рисунки](../src/gaussian_projection/proton_pilot/conditional_baseline/DECODER_ANALYSIS.md) · [показатели и команды](../src/gaussian_projection/proton_pilot/conditional_baseline/RESULTS.md) |
| Методы и воспроизводимость | [Протокол](../docs/experiment_protocol.md) · [границы применимости](../docs/experiment_state.md) · [команды](../PIPELINE.md) |

В каждом научном отчёте есть раздел о полной сумме и сигналах модулей.
[Гистограммы, квантили и доли нулей по всем 54 модулям](../src/gaussian_projection/energy_response.json).

`make reproduce` собирает офлайн-HTML с рисунками рядом с исходниками.
`make reports-pdf` обновляет три отдельных PDF и сводный файл.
`make report-bundle` собирает сводный из готовых PDF. Для объединения
нужен `pypdf` из [requirements-reports.txt](../requirements-reports.txt).

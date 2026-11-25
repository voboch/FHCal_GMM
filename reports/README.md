# Отчёты FHCal

[Три ранних отчёта о подгонке смеси и декодировании, единый PDF с закладками](FHCal_Reports_Current.pdf).

[Сводная презентация по одночастичному отклику FHCal (PDF)](overview/fhcal.pdf) · [HTML с заметками докладчика](overview/fhcal.html) · [исходник Quarto](overview/fhcal.qmd).

| Эксперимент | Материалы |
| --- | --- |
| Мюонный пилот аппроксимации отклика | [PDF](FHCal_Gaussian_Mixture_Pilot.pdf) · [Markdown и рисунки](../src/gaussian_projection/RESULTS.md) |
| Протонный пилот, число компонент и занятость | [PDF](FHCal_Proton_Projection_Pilot.pdf) · [Markdown и рисунки](../src/gaussian_projection/proton_pilot/RESULTS.md) |
| Условный банк и стохастический декодер | [PDF](FHCal_Stochastic_Decoder_Report.pdf) · [анализ и рисунки](../src/gaussian_projection/proton_pilot/conditional_baseline/DECODER_ANALYSIS.md) · [показатели и команды](../src/gaussian_projection/proton_pilot/conditional_baseline/RESULTS.md) |
| Хвосты протонного отклика по пяти направлениям | [Атлас всех модулей, слоёв и полной суммы](tails/proton_tail_atlas.pdf) · [таблица](tails/proton_tail_counts.csv) · [рецензия](../docs/tail_review_v1.md) · [план экспериментов](../docs/tail_experiment_plan.md) |
| Одночастичные хвосты и условная генерация на новых строках архива | [Результаты по модулям и условиям](single_particle/RESULTS.md) · [протокол и требования к данным](../docs/single_particle_protocol.md) |
| Парное сравнение формы при известной сумме, µ⁺ и p | [Обученные WGAN-GP и эмпирическая форма](conditional_shapes/RESULTS.md) · [протокол](conditional_shapes/PROTOCOL.md) |
| Генерация суммарного сигнала и секционных долей, µ⁺ и p | [Сравнение трёх генераторов формы при общем CFM](generated_total/RESULTS.md) · [протокол](generated_total/PROTOCOL.md) · [численный экспорт](generated_total/summary.json) |
| Отбор положительных секций при заданной сумме, µ⁺ и p | [Проверка числа секций и их локализации](section_sparsity/RESULTS.md) · [протокол](section_sparsity/PROTOCOL.md) · [численный экспорт](section_sparsity/summary.json) |
| Повторная проверка генерации суммы и формы на других архивных строках | [Результаты](generated_total_replication/RESULTS.md) · [протокол](generated_total_replication/PROTOCOL.md) · [численный экспорт](generated_total_replication/summary.json) |
| Методы и воспроизводимость | [Протокол](../docs/experiment_protocol.md) · [границы применимости](../docs/experiment_state.md) · [команды](../PIPELINE.md) |

Для исходного пилота доступны распределения полной суммы и модульных сигналов.
[Гистограммы, квантили и доли нулей по всем 54 модулям](../src/gaussian_projection/energy_response.json).

`make reproduce` собирает офлайн-HTML с рисунками рядом с исходниками.
`make reports-pdf` обновляет три отдельных PDF и сводный файл.
`make report-bundle` собирает сводный из готовых PDF. Для объединения
нужен `pypdf` из [requirements-reports.txt](../requirements-reports.txt).

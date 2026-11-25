# HIC: сохранённые протоколы парных сравнений

Перенесены из прежнего `20-result-representation.md`, §11a, без пересмотра
статистической процедуры. Применять к соответствующим дизайнам HIC-NeuralFusion,
для которых установлены описанные парность и источники вариации. Это правило
проекта, а не исчерпывающая классификация всех сравнений. Проверка фактических
источников, весов, семян и статуса результата остаётся обязательной.

Для оценки допуска действует утверждённый протокол конкретного эксперимента.
Для текущего HIC см. [поправку к правилу отбора](../../../HIC-NeuralFusion/docs/selection_rule_amendment2_20261029.md)
и [окончательное утверждение](../../../HIC-NeuralFusion/configs/selection_rule_final_approval_20261110.json).
Используйте зарегистрированные estimand, единицу сравнения, хвост интервала
и пороги. Историческое двустороннее критическое значение из этого профиля
не применяется к зарегистрированной односторонней границе, критериям KS/W/K
или парному бутстрэпу по событиям. Статус независимой оценки H задаёт
протокол проекта.

## Разные веса: две модели или переобученный преемник

Pair over the **shared training seeds**, quote the **SE of the mean** of the
per-seed differences, and give t with its degrees of freedom and the critical
value. At three seeds the original protocol uses t on 2 df with
t_crit = 4.303. Report an unresolved difference as unresolved.

## Те же веса, изменённая настройка оценки

Pair **by evaluation seed** and judge against the **evaluation-seed (repeat)
floor**. Name the changed setting, confirm that the weights are the same,
and retain the pairing. Do not apply this floor to a different-weights
comparison, and do not replace this matched evaluation with the training-seed
test from the previous case.
The repeat floor measures same-weight repeatability; keep it distinct from
the matched-event bootstrap and variation across trained seeds.

В столбце сравнения назвать применённый случай. Эти правила не разрешают
сравнивать разные populations/windows/preprocessing как одно изменение.
Если дизайн не соответствует случаям выше, использовать отдельно обоснованный
проектный протокол. Переход к иной статистической процедуре — методологическое
изменение с проверкой по §10/§30, не редакционное сокращение инструкции.

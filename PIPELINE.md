# Воспроизведение FHCal

## Генерация и отчёты из готового банка

Из корня репозитория с установленными зависимостями запустите

```sh
python inference.py -cn=proton_inference inferencer.events=100 inferencer.direction=3
make reproduce
```

Генерация использует включённый банк и калибровку. `make reproduce`
перестраивает графики и три офлайн-отчёта с встроенными PNG из
канонических JSON. Для сборки HTML нужен Quarto. Полный комплект
PDF в `reports/` обновляется следующим образом.

```sh
python -m pip install -r requirements-reports.txt
make reports-pdf
```

Для трёх исходных PDF дополнительно нужны XeLaTeX, Arial и Courier
New. `make report-bundle` собирает [единый PDF](reports/FHCal_Reports_Current.pdf)
из уже готовых отчётов. Порядок — мюонный пилот, протонный пилот,
декодер. Сборка сохраняет ссылки и добавляет закладки к разделам.

## Исходный архив для повторной подгонки

Для подгонки нужен экспорт `fhcal_data_side_modules81619_180.npz` из
архива `virtual_grid-gnn` на `200744156f5eaeeca9da503b53232febf3186cca`.
Получите файл у владельца данных или извлеките его из полного
исследовательского репозитория FHCal с доступом к этой ревизии.
В примере исходный репозиторий находится в соседнем каталоге `fhcal-source`.

```sh
export FHCAL_SOURCE=../fhcal-source
mkdir -p data
git -C "$FHCAL_SOURCE" show 2007441:GAN_project/data/fhcal_data_side_modules81619_180.zip > data/proton_archive.zip
unzip -p data/proton_archive.zip fhcal_data_side_modules81619_180.npz > data/fhcal_data_side_modules81619_180.npz
python train.py -cn=proton data.path=data/fhcal_data_side_modules81619_180.npz trainer.workers=4
```

Ожидаемый SHA-256 NPZ —
`3104573217991bd1b5be022f6cca168da1c88a9daec8b9d283539c47ee0ae130`.
Подгонка проверяет источник, карту и исключённые строки. В
`outputs/proton_bank/` сохраняются `bank.npz`, `results.json`,
`stochastic_decoder.json` и `config.yaml`. Для генерации из этого банка
добавьте `model.bank=outputs/proton_bank/bank.npz` к `inference.py`.

## Повтор чисел для стохастического декодера

```sh
mkdir -p outputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python src/analyze_gmm_packet_decoder.py data/fhcal_data_side_modules81619_180.npz \
  --output outputs/decoder_analysis.json
```

Канонический экспорт сохранён в `src/gaussian_projection/proton_pilot/conditional_baseline/`.
Для мюонного и протонного пилотов доступны
[исходные команды](src/gaussian_projection/README.md) и
[протокол повторов](docs/experiment_protocol.md).

## Пересчитать энергию модулей и полную сумму

Используйте протонный NPZ из раздела выше и мюонный NPZ из того же
исходного коммита. Для воспроизведения графиков достаточно готового
`src/gaussian_projection/energy_response.json` из репозитория.

```sh
git -C "$FHCAL_SOURCE" show 2007441:GAN_project/data/fhcal_data_muons_81619.zip > data/muon_archive.zip
unzip -p data/muon_archive.zip fhcal_data_muons_81619.npz > data/fhcal_data_muons_81619.npz
mkdir -p outputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python src/analyze_module_energy.py --protons data/fhcal_data_side_modules81619_180.npz \
  --muons data/fhcal_data_muons_81619.npz --output outputs/energy_response.json
```

## Проверить вес поглотителя

`LayeredSectionProjector` интегрирует по активным пластинам. Аргумент
`mass_depth=True` включает накопленную массовую глубину взамен линейной.
Длины и плотности задаются в `SamplingStack`. Для повторения сравнения
используйте протонный архив, тот же банк и отдельный файл вывода.

```sh
mkdir -p outputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python src/evaluate_layered_projection.py data/fhcal_data_side_modules81619_180.npz \
  --workers 4 --output outputs/layered_projection.json
```

[Укладка, сравнение и ограничения](src/gaussian_projection/proton_pilot/conditional_baseline/DECODER_ANALYSIS.md#вес-поглотителя-в-продольной-координате).
Поле `fit_telemetry` сохраняет параметры, выбранный старт и
причины остановки для активных пластин в двух координатах.

## Проверки на CPU

```sh
python -m pip install -r requirements-dev.txt
make test
```

Локальная проверка использовала Python 3.13.9, NumPy 2.4.4, SciPy 1.17.0,
Matplotlib 3.10.9 и PyYAML 6.0.3. Полная повторная подгонка дала тот же
`bank.npz` побайтово и те же метрики и выбор 128 порций. В новых квитанциях
меняются отметки версии кода и хеши происхождения. Исторические JSON
сохранены в том виде, в котором на них ссылаются отчёты.

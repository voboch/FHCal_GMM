PY ?= python
QUARTO ?= quarto
PILOT := src/gaussian_projection
PROTON := $(PILOT)/proton_pilot
DECODER := $(PROTON)/conditional_baseline
HTML_FLAGS := --fail-if-warnings --from=markdown-implicit_figures --to=html5 --standalone \
	--embed-resources --mathml --metadata lang=ru --css=$(PILOT)/report.css

.PHONY: test reproduce figures report proton-report decoder-report reports-pdf report-bundle

test:
	$(PY) -m pytest -q

figures:
	$(PY) src/plot_gaussian_projection.py
	$(PY) src/plot_gaussian_projection.py --pilot-dir $(PROTON)
	$(PY) src/plot_gmm_packet_decoder.py
	$(PY) src/plot_module_energy.py

report: figures
	$(QUARTO) pandoc $(HTML_FLAGS) --metadata pagetitle='Гауссовский пилот FHCal' \
		--resource-path=$(PILOT) --output=$(PILOT)/report.html $(PILOT)/RESULTS.md

proton-report: figures
	$(QUARTO) pandoc $(HTML_FLAGS) --metadata pagetitle='Протонный пилот FHCal' \
		--resource-path=$(PROTON) --output=$(PROTON)/report.html $(PROTON)/RESULTS.md

decoder-report: figures
	$(QUARTO) pandoc $(HTML_FLAGS) --metadata pagetitle='Стохастический декодер FHCal' \
		--resource-path=$(DECODER) --output=$(DECODER)/decoder_report.html $(DECODER)/DECODER_ANALYSIS.md

reproduce: report proton-report decoder-report

reports-pdf: figures
	$(PY) src/trainer/render_reports.py --quarto "$(QUARTO)"
	$(PY) src/trainer/bundle_reports.py

report-bundle:
	$(PY) src/trainer/bundle_reports.py

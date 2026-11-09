# Расчёты и рисунки статьи. Запуск из корня репозитория: make <цель>
PY ?= python3

.PHONY: quick experiments figures numbers verify all

quick:          ## ≈ 10 с: шесть прогонов заново и сравнение с results/main.json
	cd sim && $(PY) check.py quick

experiments:    ## все серии заново -> results/*.json (≈ 20 мин на 2 ядрах)
	cd sim && $(PY) experiments.py
	cd results && sha256sum *.json > SHA256SUMS.new && diff -q SHA256SUMS SHA256SUMS.new && rm SHA256SUMS.new || true

figures:        ## рисунки 1–4 -> figures/*.png, *.pdf
	cd sim && $(PY) figures.py

numbers:        ## все числа, приведённые в тексте статьи
	cd sim && $(PY) article_numbers.py

verify:         ## SHA-256 файлов results/*.json против results/SHA256SUMS
	cd sim && $(PY) check.py verify

all: experiments figures numbers verify

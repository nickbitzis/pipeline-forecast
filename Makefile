PORT ?= 8854

.PHONY: install test lint format run eval gate screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run --python 3.12 --group dev pytest -q

lint:
	uvx ruff check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:  ## builds the engine on first start (about 20 seconds), then loads data/forecaster.pkl
	PORT=$(PORT) uv run pipecast serve

eval:  ## full backtest, writes reports/ and caches the engine for the server
	uv run pipecast eval --save data/forecaster.pkl

gate:  ## exit 1 unless interval coverage is in band and the deal model beats the rep roll-up mid-quarter
	uv run pipecast eval --gate --out reports-ci

screenshot:  ## needs the server running on $(PORT) and Google Chrome installed (macOS path)
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
		--virtual-time-budget=10000 --window-size=1440,2900 --screenshot=docs/screenshot.png \
		"http://127.0.0.1:$(PORT)/"

docker-build:
	docker build -t pipeline-forecast .

clean:
	rm -rf .pytest_cache .ruff_cache reports-ci data

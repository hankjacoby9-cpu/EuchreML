CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
CPPFLAGS ?= -Iinclude

.PHONY: all run test test-python evaluate train-evolution benchmark benchmark-python clean

all: build/euchre_demo

ENGINE_SOURCES = src/euchre.c src/euchre_policy.c src/euchre_sim.c src/euchre_env.c \
	src/euchre_encode.c src/euchre_bridge.c
ENGINE_HEADERS = include/euchre.h include/euchre_policy.h include/euchre_sim.h \
	include/euchre_env.h include/euchre_encode.h include/euchre_bridge.h

build/euchre_demo: $(ENGINE_SOURCES) src/main.c $(ENGINE_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ENGINE_SOURCES) src/main.c -o $@

build/test_euchre: $(ENGINE_SOURCES) tests/test_euchre.c $(ENGINE_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ENGINE_SOURCES) tests/test_euchre.c -o $@

build/euchre_benchmark: $(ENGINE_SOURCES) bench/benchmark.c $(ENGINE_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ENGINE_SOURCES) bench/benchmark.c -o $@

run: build/euchre_demo
	./build/euchre_demo

test: build/test_euchre
	./build/test_euchre

test-python:
	.venv/bin/python -m unittest tests/test_python_bridge.py tests/test_evaluation.py \
		tests/test_evolution.py

evaluate:
	.venv/bin/python scripts/evaluate_policies.py

train-evolution:
	.venv/bin/python scripts/train_evolution.py

benchmark: build/euchre_benchmark
	./build/euchre_benchmark

benchmark-python:
	.venv/bin/python bench/bridge_benchmark.py

clean:
	$(RM) -r build

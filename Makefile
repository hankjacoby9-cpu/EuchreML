CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
CPPFLAGS ?= -Iinclude

.PHONY: all run test clean

all: build/euchre_demo

ENGINE_SOURCES = src/euchre.c src/euchre_sim.c
ENGINE_HEADERS = include/euchre.h include/euchre_sim.h

build/euchre_demo: $(ENGINE_SOURCES) src/main.c $(ENGINE_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ENGINE_SOURCES) src/main.c -o $@

build/test_euchre: $(ENGINE_SOURCES) tests/test_euchre.c $(ENGINE_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(ENGINE_SOURCES) tests/test_euchre.c -o $@

run: build/euchre_demo
	./build/euchre_demo

test: build/test_euchre
	./build/test_euchre

clean:
	$(RM) -r build

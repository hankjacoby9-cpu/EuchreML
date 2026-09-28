CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
CPPFLAGS ?= -Iinclude

.PHONY: all run clean

all: build/euchre_demo

build/euchre_demo: src/euchre.c src/main.c include/euchre.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(CPPFLAGS) src/euchre.c src/main.c -o $@

run: build/euchre_demo
	./build/euchre_demo

clean:
	$(RM) -r build

CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O2
LDLIBS ?= -lm

SRC = crc32.c frame.c ack.c jev_probe.c
BUILD = build

.PHONY: all test clean

all: test

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_frame: test_frame.c $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -I. -o $@ test_frame.c $(SRC) $(LDLIBS)

test: $(BUILD)/test_frame
	./$(BUILD)/test_frame

clean:
	rm -rf $(BUILD)

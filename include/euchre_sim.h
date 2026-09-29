#ifndef EUCHRE_SIM_H
#define EUCHRE_SIM_H

#include "euchre.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Deal and play one hand using the project's simple baseline strategy.
 * bot_state keeps random card choices reproducible across an entire match.
 */
bool euchre_play_simple_hand(EuchreGame *game, uint64_t *bot_state);

#endif

#ifndef EUCHRE_BRIDGE_H
#define EUCHRE_BRIDGE_H

#include "euchre_encode.h"
#include "euchre_env.h"

#include <stdint.h>

/* Opaque lifecycle functions exported to foreign-language bindings. */
EuchreEnv *euchre_bridge_create(void);
void euchre_bridge_destroy(EuchreEnv *env);

/* Reset and step directly into the frozen numeric buffers. */
int euchre_bridge_reset(
    EuchreEnv *env, uint64_t seed, int learning_seat,
    int16_t observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t action_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward);

int euchre_bridge_step(
    EuchreEnv *env, int action,
    int16_t observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t action_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward);

/* Runtime queries let bindings reject incompatible compiled libraries. */
int euchre_bridge_layout_version(void);
int euchre_bridge_observation_size(void);
int euchre_bridge_action_count(void);

#endif

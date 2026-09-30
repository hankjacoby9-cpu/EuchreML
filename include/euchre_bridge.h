#ifndef EUCHRE_BRIDGE_H
#define EUCHRE_BRIDGE_H

#include "euchre_encode.h"
#include "euchre_env.h"

#include <stdint.h>
#include <stddef.h>

typedef struct EuchreBatchEnv EuchreBatchEnv;

/* Opaque lifecycle functions exported to foreign-language bindings. */
EuchreEnv *euchre_bridge_create(void);
void euchre_bridge_destroy(EuchreEnv *env);

/* Reset and step directly into the frozen numeric buffers. */
int euchre_bridge_reset(
    EuchreEnv *env, uint64_t seed, int learning_seat,
    int16_t observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t action_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward);

int euchre_bridge_reset_team(
    EuchreEnv *env, uint64_t seed, int controlled_team,
    int16_t observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t action_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward);

int euchre_bridge_step(
    EuchreEnv *env, int action,
    int16_t observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t action_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward);

/* Own and advance several environments through one foreign-function call. */
EuchreBatchEnv *euchre_bridge_batch_create(size_t environment_count);
void euchre_bridge_batch_destroy(EuchreBatchEnv *batch);
size_t euchre_bridge_batch_size(const EuchreBatchEnv *batch);

int euchre_bridge_batch_reset(
    EuchreBatchEnv *batch, const uint64_t seeds[], const int learning_seats[],
    int16_t observations[], uint8_t action_masks[], int rewards[],
    int statuses[]);

int euchre_bridge_batch_reset_teams(
    EuchreBatchEnv *batch, const uint64_t seeds[], const int teams[],
    int16_t observations[], uint8_t action_masks[], int rewards[],
    int statuses[]);

int euchre_bridge_batch_step(
    EuchreBatchEnv *batch, const int actions[], int16_t observations[],
    uint8_t action_masks[], int rewards[], int statuses[]);

/* Reset selected terminal slots while stepping the remaining active slots. */
int euchre_bridge_batch_advance(
    EuchreBatchEnv *batch, const int actions[], const uint8_t reset_flags[],
    const uint64_t seeds[], const int learning_seats[], int16_t observations[],
    uint8_t action_masks[], int rewards[], int statuses[]);

int euchre_bridge_batch_advance_teams(
    EuchreBatchEnv *batch, const int actions[], const uint8_t reset_flags[],
    const uint64_t seeds[], const int teams[], int16_t observations[],
    uint8_t action_masks[], int rewards[], int statuses[]);

/* Runtime queries let bindings reject incompatible compiled libraries. */
int euchre_bridge_layout_version(void);
int euchre_bridge_observation_size(void);
int euchre_bridge_action_count(void);

#endif

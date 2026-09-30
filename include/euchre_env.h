#ifndef EUCHRE_ENV_H
#define EUCHRE_ENV_H

#include "euchre_policy.h"

#include <stdint.h>

typedef struct EuchreEnv EuchreEnv;

typedef enum {
    EUCHRE_ENV_ERROR = -1,
    EUCHRE_ENV_DECISION,
    EUCHRE_ENV_TERMINAL
} EuchreEnvStatus;

/* Allocate and release one opaque single-hand learning environment. */
EuchreEnv *euchre_env_create(void);
void euchre_env_destroy(EuchreEnv *env);

/*
 * Start a hand and advance until the learning seat has a meaningful choice.
 * Forced actions and all opponent decisions are applied inside C.
 */
EuchreEnvStatus euchre_env_reset(
    EuchreEnv *env, uint64_t seed, int learning_seat,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

EuchreEnvStatus euchre_env_reset_with_dealer(
    EuchreEnv *env, uint64_t seed, int learning_seat, int dealer,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

/* Control both seats on one partnership while observing only the acting seat. */
EuchreEnvStatus euchre_env_reset_team(
    EuchreEnv *env, uint64_t seed, int controlled_team,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

EuchreEnvStatus euchre_env_reset_team_with_dealer(
    EuchreEnv *env, uint64_t seed, int controlled_team, int dealer,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

/*
 * Apply one learning-seat action and advance to its next meaningful decision.
 * Reward remains zero until terminal, then equals team points minus opponents'.
 */
EuchreEnvStatus euchre_env_step(
    EuchreEnv *env, EuchreAction action, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

#endif

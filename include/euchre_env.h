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

/* Cap both scores at the target before calculating one team's final margin. */
bool euchre_match_reward(int score0, int score1, int team, int target_score,
                         int *reward);

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

/* Play consecutive hands to target_score with both partnership seats controlled. */
EuchreEnvStatus euchre_env_reset_team_match(
    EuchreEnv *env, uint64_t seed, int controlled_team, int starting_dealer,
    int target_score, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

/* Expose every non-forced decision, one acting seat's private view at a time. */
EuchreEnvStatus euchre_env_reset_head_to_head_match(
    EuchreEnv *env, uint64_t seed, int reward_team, int starting_dealer,
    int target_score, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

/*
 * Apply one learning-seat action and advance to its next meaningful decision.
 * Reward remains zero until terminal, then equals team points minus opponents'.
 */
EuchreEnvStatus euchre_env_step(
    EuchreEnv *env, EuchreAction action, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward);

#endif

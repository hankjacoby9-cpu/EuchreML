#include "euchre_env.h"

#include "euchre_sim.h"

#include <stdlib.h>
#include <string.h>

#define MAX_ENV_ACTIONS_PER_HAND 64

struct EuchreEnv {
    EuchreGame game;
    unsigned controlled_seats;
    int reward_team;
    int observation_seat;
    uint64_t opponent_state;
    bool active;
};

/* Return the only legal action when the current state has no real choice. */
static EuchreAction only_legal_action(
    const bool legal_actions[EUCHRE_ACTION_COUNT]) {
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        if (legal_actions[action]) return (EuchreAction)action;
    }
    return EUCHRE_ACTION_INVALID;
}

/* Fill terminal output and calculate reward from the learning team's view. */
static EuchreEnvStatus finish_hand(
    EuchreEnv *env, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    *reward = env->game.score[env->reward_team] -
              env->game.score[1 - env->reward_team];
    memset(legal_actions, 0, sizeof(bool) * EUCHRE_ACTION_COUNT);
    if (!euchre_observe(&env->game, env->observation_seat, observation)) {
        return EUCHRE_ENV_ERROR;
    }
    env->active = false;
    return EUCHRE_ENV_TERMINAL;
}

/* Run forced moves and opponents until learning can choose or the hand ends. */
static EuchreEnvStatus advance_to_decision(
    EuchreEnv *env, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    *reward = 0;
    for (int action_number = 0; action_number < MAX_ENV_ACTIONS_PER_HAND;
         ++action_number) {
        if (env->game.phase == EUCHRE_HAND_COMPLETE) {
            return finish_hand(env, observation, legal_actions, reward);
        }

        size_t legal_count =
            euchre_policy_legal_actions(&env->game, legal_actions);
        if (legal_count == 0) return EUCHRE_ENV_ERROR;

        int player = env->game.current_player;
        if ((env->controlled_seats & (1U << player)) && legal_count > 1) {
            env->observation_seat = player;
            if (!euchre_observe(&env->game, player, observation)) {
                return EUCHRE_ENV_ERROR;
            }
            return EUCHRE_ENV_DECISION;
        }

        EuchreAction action;
        if (legal_count == 1) {
            action = only_legal_action(legal_actions);
        } else {
            EuchreObservation opponent_view;
            if (!euchre_observe(&env->game, player, &opponent_view)) {
                return EUCHRE_ENV_ERROR;
            }
            action = euchre_simple_policy(&opponent_view, legal_actions,
                                          &env->opponent_state);
        }
        if (!euchre_apply_action(&env->game, action)) {
            return EUCHRE_ENV_ERROR;
        }
    }
    return EUCHRE_ENV_ERROR;
}

/* Allocate one environment without exposing its internal game state. */
EuchreEnv *euchre_env_create(void) {
    return calloc(1, sizeof(EuchreEnv));
}

/* Release an environment created by euchre_env_create. */
void euchre_env_destroy(EuchreEnv *env) {
    free(env);
}

/* Start a new single-hand episode and advance to its first real decision. */
EuchreEnvStatus euchre_env_reset(
    EuchreEnv *env, uint64_t seed, int learning_seat,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    return euchre_env_reset_with_dealer(
        env, seed, learning_seat, 0, observation, legal_actions, reward);
}

EuchreEnvStatus euchre_env_reset_with_dealer(
    EuchreEnv *env, uint64_t seed, int learning_seat, int dealer,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    if (env == NULL || observation == NULL || legal_actions == NULL ||
        reward == NULL || learning_seat < 0 || learning_seat >= EUCHRE_PLAYERS) {
        return EUCHRE_ENV_ERROR;
    }
    if (dealer < 0 || dealer >= EUCHRE_PLAYERS) return EUCHRE_ENV_ERROR;

    euchre_init(&env->game, seed);
    env->controlled_seats = 1U << learning_seat;
    env->reward_team = learning_seat % 2;
    env->observation_seat = learning_seat;
    env->opponent_state = seed ^ UINT64_C(0xa0761d6478bd642f);
    env->active = true;
    env->game.dealer = (dealer + EUCHRE_PLAYERS - 1) % EUCHRE_PLAYERS;
    euchre_deal(&env->game);

    return advance_to_decision(env, observation, legal_actions, reward);
}

EuchreEnvStatus euchre_env_reset_team(
    EuchreEnv *env, uint64_t seed, int controlled_team,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    return euchre_env_reset_team_with_dealer(
        env, seed, controlled_team, 0, observation, legal_actions, reward);
}

EuchreEnvStatus euchre_env_reset_team_with_dealer(
    EuchreEnv *env, uint64_t seed, int controlled_team, int dealer,
    EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    if (env == NULL || observation == NULL || legal_actions == NULL ||
        reward == NULL || controlled_team < 0 || controlled_team > 1) {
        return EUCHRE_ENV_ERROR;
    }
    if (dealer < 0 || dealer >= EUCHRE_PLAYERS) return EUCHRE_ENV_ERROR;

    euchre_init(&env->game, seed);
    env->controlled_seats = (1U << controlled_team) |
                            (1U << (controlled_team + 2));
    env->reward_team = controlled_team;
    env->observation_seat = controlled_team;
    env->opponent_state = seed ^ UINT64_C(0xa0761d6478bd642f);
    env->active = true;
    env->game.dealer = (dealer + EUCHRE_PLAYERS - 1) % EUCHRE_PLAYERS;
    euchre_deal(&env->game);

    return advance_to_decision(env, observation, legal_actions, reward);
}

/* Apply a meaningful learning action and advance to its next decision. */
EuchreEnvStatus euchre_env_step(
    EuchreEnv *env, EuchreAction action, EuchreObservation *observation,
    bool legal_actions[EUCHRE_ACTION_COUNT], int *reward) {
    if (env == NULL || observation == NULL || legal_actions == NULL ||
        reward == NULL || !env->active ||
        !(env->controlled_seats & (1U << env->game.current_player))) {
        return EUCHRE_ENV_ERROR;
    }

    bool current_legal[EUCHRE_ACTION_COUNT];
    size_t legal_count =
        euchre_policy_legal_actions(&env->game, current_legal);
    if (legal_count <= 1 || action < 0 || action >= EUCHRE_ACTION_COUNT ||
        !current_legal[action] || !euchre_apply_action(&env->game, action)) {
        return EUCHRE_ENV_ERROR;
    }
    return advance_to_decision(env, observation, legal_actions, reward);
}

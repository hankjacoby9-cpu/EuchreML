#include "euchre_bridge.h"

/* Encode one environment response only after the engine action succeeds. */
static int encode_result(
    int status, const EuchreObservation *observation,
    const bool legal_actions[EUCHRE_ACTION_COUNT], int16_t output_observation[],
    uint8_t output_mask[]) {
    if (status == EUCHRE_ENV_ERROR ||
        !euchre_encode_observation(observation, output_observation) ||
        !euchre_encode_action_mask(legal_actions, output_mask)) {
        return EUCHRE_ENV_ERROR;
    }
    return status;
}

EuchreEnv *euchre_bridge_create(void) {
    return euchre_env_create();
}

void euchre_bridge_destroy(EuchreEnv *env) {
    euchre_env_destroy(env);
}

int euchre_bridge_reset(
    EuchreEnv *env, uint64_t seed, int learning_seat,
    int16_t output_observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t output_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward) {
    if (output_observation == NULL || output_mask == NULL || reward == NULL) {
        return EUCHRE_ENV_ERROR;
    }
    EuchreObservation observation;
    bool legal_actions[EUCHRE_ACTION_COUNT];
    int status = euchre_env_reset(env, seed, learning_seat, &observation,
                                  legal_actions, reward);
    return encode_result(status, &observation, legal_actions,
                         output_observation, output_mask);
}

int euchre_bridge_step(
    EuchreEnv *env, int action,
    int16_t output_observation[EUCHRE_OBSERVATION_SIZE],
    uint8_t output_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE], int *reward) {
    if (output_observation == NULL || output_mask == NULL || reward == NULL) {
        return EUCHRE_ENV_ERROR;
    }
    EuchreObservation observation;
    bool legal_actions[EUCHRE_ACTION_COUNT];
    int status = euchre_env_step(env, (EuchreAction)action, &observation,
                                 legal_actions, reward);
    return encode_result(status, &observation, legal_actions,
                         output_observation, output_mask);
}

int euchre_bridge_layout_version(void) {
    return EUCHRE_OBSERVATION_LAYOUT_VERSION;
}

int euchre_bridge_observation_size(void) {
    return EUCHRE_OBSERVATION_SIZE;
}

int euchre_bridge_action_count(void) {
    return EUCHRE_ACTION_COUNT;
}

#include "euchre_bridge.h"

#include <stdlib.h>

struct EuchreBatchEnv {
    size_t size;
    EuchreEnv **envs;
    int *statuses;
};

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

EuchreBatchEnv *euchre_bridge_batch_create(size_t environment_count) {
    if (environment_count == 0) return NULL;

    EuchreBatchEnv *batch = calloc(1, sizeof(*batch));
    if (batch == NULL) return NULL;
    batch->envs = calloc(environment_count, sizeof(*batch->envs));
    batch->statuses = calloc(environment_count, sizeof(*batch->statuses));
    if (batch->envs == NULL || batch->statuses == NULL) {
        euchre_bridge_batch_destroy(batch);
        return NULL;
    }

    batch->size = environment_count;
    for (size_t index = 0; index < environment_count; ++index) {
        batch->envs[index] = euchre_env_create();
        if (batch->envs[index] == NULL) {
            euchre_bridge_batch_destroy(batch);
            return NULL;
        }
        batch->statuses[index] = EUCHRE_ENV_ERROR;
    }
    return batch;
}

void euchre_bridge_batch_destroy(EuchreBatchEnv *batch) {
    if (batch == NULL) return;
    if (batch->envs != NULL) {
        for (size_t index = 0; index < batch->size; ++index) {
            euchre_env_destroy(batch->envs[index]);
        }
    }
    free(batch->statuses);
    free(batch->envs);
    free(batch);
}

size_t euchre_bridge_batch_size(const EuchreBatchEnv *batch) {
    return batch == NULL ? 0 : batch->size;
}

int euchre_bridge_batch_reset(
    EuchreBatchEnv *batch, const uint64_t seeds[], const int learning_seats[],
    int16_t observations[], uint8_t action_masks[], int rewards[],
    int statuses[]) {
    if (batch == NULL || seeds == NULL || learning_seats == NULL ||
        observations == NULL || action_masks == NULL || rewards == NULL ||
        statuses == NULL) {
        return 0;
    }

    int success = 1;
    for (size_t index = 0; index < batch->size; ++index) {
        int status = euchre_bridge_reset(
            batch->envs[index], seeds[index], learning_seats[index],
            observations + index * EUCHRE_OBSERVATION_SIZE,
            action_masks + index * EUCHRE_ENCODED_ACTION_MASK_SIZE,
            rewards + index);
        batch->statuses[index] = status;
        statuses[index] = status;
        if (status == EUCHRE_ENV_ERROR) success = 0;
    }
    return success;
}

int euchre_bridge_batch_step(
    EuchreBatchEnv *batch, const int actions[], int16_t observations[],
    uint8_t action_masks[], int rewards[], int statuses[]) {
    if (batch == NULL || actions == NULL || observations == NULL ||
        action_masks == NULL || rewards == NULL || statuses == NULL) {
        return 0;
    }

    int success = 1;
    for (size_t index = 0; index < batch->size; ++index) {
        int status = batch->statuses[index];
        if (status == EUCHRE_ENV_DECISION) {
            status = euchre_bridge_step(
                batch->envs[index], actions[index],
                observations + index * EUCHRE_OBSERVATION_SIZE,
                action_masks + index * EUCHRE_ENCODED_ACTION_MASK_SIZE,
                rewards + index);
            batch->statuses[index] = status;
        }
        statuses[index] = status;
        if (status == EUCHRE_ENV_ERROR) success = 0;
    }
    return success;
}

int euchre_bridge_batch_advance(
    EuchreBatchEnv *batch, const int actions[], const uint8_t reset_flags[],
    const uint64_t seeds[], const int learning_seats[], int16_t observations[],
    uint8_t action_masks[], int rewards[], int statuses[]) {
    if (batch == NULL || actions == NULL || reset_flags == NULL ||
        seeds == NULL || learning_seats == NULL || observations == NULL ||
        action_masks == NULL || rewards == NULL || statuses == NULL) {
        return 0;
    }

    int success = 1;
    for (size_t index = 0; index < batch->size; ++index) {
        int status = batch->statuses[index];
        if (reset_flags[index]) {
            status = euchre_bridge_reset(
                batch->envs[index], seeds[index], learning_seats[index],
                observations + index * EUCHRE_OBSERVATION_SIZE,
                action_masks + index * EUCHRE_ENCODED_ACTION_MASK_SIZE,
                rewards + index);
        } else if (status == EUCHRE_ENV_DECISION) {
            status = euchre_bridge_step(
                batch->envs[index], actions[index],
                observations + index * EUCHRE_OBSERVATION_SIZE,
                action_masks + index * EUCHRE_ENCODED_ACTION_MASK_SIZE,
                rewards + index);
        }
        batch->statuses[index] = status;
        statuses[index] = status;
        if (status == EUCHRE_ENV_ERROR) success = 0;
    }
    return success;
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

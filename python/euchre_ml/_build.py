from cffi import FFI


ffibuilder = FFI()

ffibuilder.cdef(
    """
    typedef struct EuchreEnv EuchreEnv;
    typedef struct EuchreBatchEnv EuchreBatchEnv;

    EuchreEnv *euchre_bridge_create(void);
    void euchre_bridge_destroy(EuchreEnv *env);

    int euchre_bridge_reset(EuchreEnv *env, uint64_t seed, int learning_seat,
                            int16_t *observation, uint8_t *action_mask,
                            int *reward);
    int euchre_bridge_step(EuchreEnv *env, int action, int16_t *observation,
                           uint8_t *action_mask, int *reward);

    EuchreBatchEnv *euchre_bridge_batch_create(size_t environment_count);
    void euchre_bridge_batch_destroy(EuchreBatchEnv *batch);
    size_t euchre_bridge_batch_size(const EuchreBatchEnv *batch);
    int euchre_bridge_batch_reset(EuchreBatchEnv *batch,
                                  const uint64_t *seeds,
                                  const int *learning_seats,
                                  int16_t *observations,
                                  uint8_t *action_masks, int *rewards,
                                  int *statuses);
    int euchre_bridge_batch_step(EuchreBatchEnv *batch, const int *actions,
                                 int16_t *observations,
                                 uint8_t *action_masks, int *rewards,
                                 int *statuses);
    int euchre_bridge_batch_advance(EuchreBatchEnv *batch,
                                    const int *actions,
                                    const uint8_t *reset_flags,
                                    const uint64_t *seeds,
                                    const int *learning_seats,
                                    int16_t *observations,
                                    uint8_t *action_masks, int *rewards,
                                    int *statuses);

    int euchre_bridge_layout_version(void);
    int euchre_bridge_observation_size(void);
    int euchre_bridge_action_count(void);
    """
)

ffibuilder.set_source(
    "euchre_ml._native",
    '#include "euchre_bridge.h"',
    include_dirs=["include"],
    sources=[
        f"src/{name}"
        for name in (
            "euchre.c",
            "euchre_policy.c",
            "euchre_sim.c",
            "euchre_env.c",
            "euchre_encode.c",
            "euchre_bridge.c",
        )
    ],
    extra_compile_args=["-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic"],
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

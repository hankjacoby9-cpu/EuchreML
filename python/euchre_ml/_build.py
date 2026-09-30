from cffi import FFI


ffibuilder = FFI()

ffibuilder.cdef(
    """
    typedef struct EuchreEnv EuchreEnv;

    EuchreEnv *euchre_bridge_create(void);
    void euchre_bridge_destroy(EuchreEnv *env);

    int euchre_bridge_reset(EuchreEnv *env, uint64_t seed, int learning_seat,
                            int16_t *observation, uint8_t *action_mask,
                            int *reward);
    int euchre_bridge_step(EuchreEnv *env, int action, int16_t *observation,
                           uint8_t *action_mask, int *reward);

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

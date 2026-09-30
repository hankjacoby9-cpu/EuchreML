#include "euchre.h"
#include "euchre_sim.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_HANDS UINT64_C(100000)

/* Run headless baseline hands and report throughput without Python overhead. */
int main(int argc, char **argv) {
    uint64_t requested = DEFAULT_HANDS;
    if (argc == 2) {
        char *end = NULL;
        unsigned long long parsed = strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || parsed == 0) {
            fprintf(stderr, "Usage: %s [positive-hand-count]\n", argv[0]);
            return 1;
        }
        requested = (uint64_t)parsed;
    }

    EuchreGame game;
    uint64_t bot_state = UINT64_C(0xa0761d6478bd642f);
    euchre_init(&game, UINT64_C(0x243f6a8885a308d3));

    clock_t start = clock();
    for (uint64_t hand = 0; hand < requested; ++hand) {
        if (!euchre_play_simple_hand(&game, &bot_state)) {
            fprintf(stderr, "Simulation failed at hand %" PRIu64 ".\n", hand);
            return 1;
        }
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    double rate = seconds > 0.0 ? (double)requested / seconds : 0.0;
    printf("Simulated %" PRIu64 " hands in %.3f seconds (%.0f hands/second).\n",
           requested, seconds, rate);
    return 0;
}

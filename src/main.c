#include "euchre.h"
#include "euchre_sim.h"

#include <stdio.h>
#include <time.h>

#define WINNING_SCORE 10

/* Play a complete demonstration match and summarize every hand. */
int main(void) {
    uint64_t seed = (uint64_t)time(NULL);
    uint64_t bot_state = seed ^ UINT64_C(0x9e3779b97f4a7c15);
    EuchreGame game;
    euchre_init(&game, seed);

    int hand_number = 0;
    while (game.score[0] < WINNING_SCORE && game.score[1] < WINNING_SCORE) {
        int previous_score[2] = {game.score[0], game.score[1]};
        if (!euchre_play_simple_hand(&game, &bot_state)) {
            fprintf(stderr, "The simulation reached an invalid game state.\n");
            return 1;
        }

        ++hand_number;
        int scoring_team = game.score[0] > previous_score[0] ? 0 : 1;
        int points = game.score[scoring_team] - previous_score[scoring_team];
        printf("Hand %2d: Player %d called %-8s%s | Team %d +%d | score %d-%d\n",
               hand_number, game.caller, euchre_suit_name(game.trump),
               game.going_alone ? " alone" : "      ", scoring_team, points,
               game.score[0], game.score[1]);
    }

    int winner = game.score[0] >= WINNING_SCORE ? 0 : 1;
    printf("\nTeam %d wins the match %d-%d after %d hands.\n", winner,
           game.score[0], game.score[1], hand_number);
    return 0;
}

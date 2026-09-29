#include "euchre.h"
#include "euchre_sim.h"

#include <stdint.h>
#include <stdio.h>

#define WINNING_SCORE 10
#define MAX_HANDS_PER_MATCH 1000

static int failures;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
            ++failures;                                                         \
            return;                                                             \
        }                                                                       \
    } while (0)

/* Confirm round two rejects the suit that was turned down in round one. */
static void test_cannot_call_turned_suit(void) {
    EuchreGame game;
    euchre_init(&game, 101);
    euchre_deal(&game);
    EuchreSuit turned_suit = game.turned_suit;

    for (int i = 0; i < EUCHRE_PLAYERS; ++i) CHECK(euchre_pass_bid(&game));
    CHECK(game.phase == EUCHRE_BIDDING_ROUND_TWO);

    int bidder = game.current_player;
    CHECK(!euchre_call_trump(&game, turned_suit, false));
    CHECK(game.phase == EUCHRE_BIDDING_ROUND_TWO);
    CHECK(game.current_player == bidder);
    CHECK(game.trump == EUCHRE_NO_SUIT);
}

/* Confirm the dealer cannot pass after the other three second-round passes. */
static void test_stick_the_dealer(void) {
    EuchreGame game;
    euchre_init(&game, 202);
    euchre_deal(&game);

    for (int i = 0; i < EUCHRE_PLAYERS; ++i) CHECK(euchre_pass_bid(&game));
    for (int i = 0; i < EUCHRE_PLAYERS - 1; ++i) CHECK(euchre_pass_bid(&game));
    CHECK(game.current_player == game.dealer);
    CHECK(!euchre_pass_bid(&game));
    CHECK(game.current_player == game.dealer);
}

/* Confirm a lone caller's partner is skipped for all five three-card tricks. */
static void test_lone_hand_turn_order(void) {
    EuchreGame game;
    euchre_init(&game, 303);
    euchre_deal(&game);

    int caller = game.current_player;
    CHECK(euchre_order_up(&game, true));
    CHECK(euchre_dealer_discard(&game, 0));
    CHECK(game.sitting_out == (caller + 2) % EUCHRE_PLAYERS);

    int plays = 0;
    while (game.phase == EUCHRE_PLAYING) {
        CHECK(game.current_player != game.sitting_out);
        size_t legal[EUCHRE_HAND_SIZE];
        size_t count = euchre_legal_moves(&game, game.current_player, legal);
        CHECK(count > 0);
        CHECK(euchre_play_card(&game, legal[0], NULL));
        ++plays;
    }

    CHECK(plays == 15);
    CHECK(game.tricks_won[0] + game.tricks_won[1] == EUCHRE_HAND_SIZE);
    CHECK(game.hands[game.sitting_out].count == EUCHRE_HAND_SIZE);
}

/* Run a complete match and check scoring and hand invariants after every deal. */
static void run_seeded_match(uint64_t seed) {
    EuchreGame game;
    uint64_t bot_state = seed ^ UINT64_C(0xa0761d6478bd642f);
    euchre_init(&game, seed);

    int hands = 0;
    while (game.score[0] < WINNING_SCORE && game.score[1] < WINNING_SCORE) {
        int before[2] = {game.score[0], game.score[1]};
        CHECK(euchre_play_simple_hand(&game, &bot_state));
        ++hands;

        CHECK(game.phase == EUCHRE_HAND_COMPLETE);
        CHECK(game.tricks_played == EUCHRE_HAND_SIZE);
        CHECK(game.tricks_won[0] + game.tricks_won[1] == EUCHRE_HAND_SIZE);

        int delta0 = game.score[0] - before[0];
        int delta1 = game.score[1] - before[1];
        CHECK((delta0 == 0) != (delta1 == 0));
        int points = delta0 != 0 ? delta0 : delta1;
        CHECK(points == 1 || points == 2 || points == 4);
        CHECK(hands < MAX_HANDS_PER_MATCH);
    }

    CHECK((game.score[0] >= WINNING_SCORE) !=
          (game.score[1] >= WINNING_SCORE));
}

/* Exercise full matches across unrelated shuffle and bot-decision seeds. */
static void test_seeded_matches(void) {
    const uint64_t seeds[] = {1, 7, 42, 2026, UINT64_C(0xdeadbeef)};
    for (size_t i = 0; i < sizeof(seeds) / sizeof(seeds[0]); ++i) {
        run_seeded_match(seeds[i]);
        if (failures != 0) return;
    }
}

int main(void) {
    test_cannot_call_turned_suit();
    test_stick_the_dealer();
    test_lone_hand_turn_order();
    test_seeded_matches();

    if (failures != 0) {
        fprintf(stderr, "%d test group(s) failed.\n", failures);
        return 1;
    }
    printf("All Euchre engine tests passed.\n");
    return 0;
}

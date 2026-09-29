#include "euchre.h"
#include "euchre_policy.h"
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

/* Count enabled entries without depending only on the mask builder's return value. */
static size_t count_mask(const bool legal[EUCHRE_ACTION_COUNT]) {
    size_t count = 0;
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        if (legal[action]) ++count;
    }
    return count;
}

/* Return the first enabled card action in a mask. */
static EuchreAction first_card_action(
    const bool legal[EUCHRE_ACTION_COUNT]) {
    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        if (legal[card_id]) return (EuchreAction)card_id;
    }
    return EUCHRE_ACTION_INVALID;
}

/* Confirm the opening-bid and dealer-discard observations and masks. */
static void test_round_one_and_discard_masks(void) {
    EuchreGame game;
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    euchre_init(&game, 99);
    euchre_deal(&game);

    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(observation.hand_count == EUCHRE_HAND_SIZE);
    CHECK(observation.player == game.current_player);
    CHECK(euchre_policy_legal_actions(&game, legal) == 3);
    CHECK(count_mask(legal) == 3);
    CHECK(legal[EUCHRE_ACTION_PASS]);
    CHECK(legal[EUCHRE_ACTION_ORDER_UP]);
    CHECK(legal[EUCHRE_ACTION_ORDER_UP_ALONE]);
    CHECK(!euchre_apply_action(&game, EUCHRE_ACTION_CALL_CLUBS));

    CHECK(euchre_apply_action(&game, EUCHRE_ACTION_ORDER_UP));
    CHECK(game.current_player == game.dealer);
    CHECK(euchre_observe(&game, game.dealer, &observation));
    CHECK(observation.hand_count == EUCHRE_HAND_SIZE + 1);
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(count_mask(legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(!legal[EUCHRE_ACTION_PASS]);
    CHECK(!legal[EUCHRE_ACTION_ORDER_UP]);
    for (size_t i = 0; i < game.hands[game.dealer].count; ++i) {
        CHECK(legal[euchre_card_id(game.hands[game.dealer].cards[i])]);
    }
}

/* Confirm ordinary and stick-the-dealer masks during second-round bidding. */
static void test_round_two_masks(void) {
    EuchreGame game;
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    euchre_init(&game, 100);
    euchre_deal(&game);

    for (int i = 0; i < EUCHRE_PLAYERS; ++i) {
        CHECK(euchre_apply_action(&game, EUCHRE_ACTION_PASS));
    }
    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(observation.phase == EUCHRE_BIDDING_ROUND_TWO);
    CHECK(observation.bid_turns == 0);
    CHECK(euchre_policy_legal_actions(&game, legal) == 7);
    CHECK(count_mask(legal) == 7);
    CHECK(legal[EUCHRE_ACTION_PASS]);

    for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
        bool expected = (EuchreSuit)suit != game.turned_suit;
        CHECK(legal[EUCHRE_ACTION_CALL_CLUBS + suit] == expected);
        CHECK(legal[EUCHRE_ACTION_CALL_CLUBS_ALONE + suit] == expected);
    }

    for (int i = 0; i < EUCHRE_PLAYERS - 1; ++i) {
        CHECK(euchre_apply_action(&game, EUCHRE_ACTION_PASS));
    }
    CHECK(game.current_player == game.dealer);
    CHECK(euchre_policy_legal_actions(&game, legal) == 6);
    CHECK(count_mask(legal) == 6);
    CHECK(!legal[EUCHRE_ACTION_PASS]);
    CHECK(!euchre_apply_action(&game, EUCHRE_ACTION_PASS));
}

/* Confirm play masks follow suit and observations update after each card. */
static void test_play_and_complete_masks(void) {
    EuchreGame game;
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    euchre_init(&game, 404);
    euchre_deal(&game);

    CHECK(euchre_apply_action(&game, EUCHRE_ACTION_ORDER_UP));
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(euchre_apply_action(&game, first_card_action(legal)));
    CHECK(game.phase == EUCHRE_PLAYING);

    int leader = game.current_player;
    CHECK(euchre_observe(&game, leader, &observation));
    CHECK(observation.phase == EUCHRE_PLAYING);
    CHECK(observation.hand_count == EUCHRE_HAND_SIZE);
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE);
    CHECK(count_mask(legal) == EUCHRE_HAND_SIZE);

    EuchreAction led_action = first_card_action(legal);
    CHECK(led_action != EUCHRE_ACTION_INVALID);
    CHECK(euchre_apply_action(&game, led_action));
    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(observation.cards_played[led_action]);
    CHECK(observation.trick_slot_used[leader]);
    CHECK(euchre_card_id(observation.trick[leader]) == led_action);

    EuchreSuit led_suit = euchre_effective_suit(game.trick[leader], game.trump);
    size_t following_cards = 0;
    const EuchreHand *hand = &game.hands[game.current_player];
    for (size_t i = 0; i < hand->count; ++i) {
        if (euchre_effective_suit(hand->cards[i], game.trump) == led_suit) {
            ++following_cards;
        }
    }
    size_t expected_legal = following_cards == 0 ? hand->count : following_cards;
    CHECK(euchre_policy_legal_actions(&game, legal) == expected_legal);
    CHECK(count_mask(legal) == expected_legal);

    while (game.phase == EUCHRE_PLAYING) {
        CHECK(euchre_policy_legal_actions(&game, legal) > 0);
        CHECK(euchre_apply_action(&game, first_card_action(legal)));
    }

    CHECK(game.phase == EUCHRE_HAND_COMPLETE);
    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(observation.phase == EUCHRE_HAND_COMPLETE);
    CHECK(euchre_policy_legal_actions(&game, legal) == 0);
    CHECK(count_mask(legal) == 0);
}

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
    test_round_one_and_discard_masks();
    test_round_two_masks();
    test_play_and_complete_masks();
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

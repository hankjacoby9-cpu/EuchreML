#include "euchre.h"
#include "euchre_bridge.h"
#include "euchre_encode.h"
#include "euchre_env.h"
#include "euchre_policy.h"
#include "euchre_sim.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

/* Return the first action of any kind enabled by a policy mask. */
static EuchreAction first_legal_action(
    const bool legal[EUCHRE_ACTION_COUNT]) {
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        if (legal[action]) return (EuchreAction)action;
    }
    return EUCHRE_ACTION_INVALID;
}

/* Run every seat through decision-point episodes and require real choices only. */
static void test_decision_point_environment(void) {
    const uint64_t seeds[] = {1, 42, 2026, UINT64_C(0xabcddcba)};
    EuchreEnv *env = euchre_env_create();
    CHECK(env != NULL);

    for (int seat = 0; seat < EUCHRE_PLAYERS; ++seat) {
        for (size_t seed_index = 0;
             seed_index < sizeof(seeds) / sizeof(seeds[0]); ++seed_index) {
            EuchreObservation observation;
            bool legal[EUCHRE_ACTION_COUNT];
            int reward = 99;
            EuchreEnvStatus status = euchre_env_reset(
                env, seeds[seed_index], seat, &observation, legal, &reward);
            CHECK(status != EUCHRE_ENV_ERROR);
            CHECK(reward == 0 || status == EUCHRE_ENV_TERMINAL);

            int decisions = 0;
            while (status == EUCHRE_ENV_DECISION) {
                CHECK(observation.player == seat);
                CHECK(observation.current_player == seat);
                CHECK(count_mask(legal) > 1);
                EuchreAction action = first_legal_action(legal);
                CHECK(action != EUCHRE_ACTION_INVALID);
                status = euchre_env_step(env, action, &observation, legal,
                                         &reward);
                CHECK(status != EUCHRE_ENV_ERROR);
                if (status == EUCHRE_ENV_DECISION) CHECK(reward == 0);
                CHECK(++decisions < MAX_HANDS_PER_MATCH);
            }

            CHECK(status == EUCHRE_ENV_TERMINAL);
            CHECK(observation.phase == EUCHRE_HAND_COMPLETE);
            CHECK(count_mask(legal) == 0);
            CHECK(reward == -4 || reward == -2 || reward == -1 ||
                  reward == 1 || reward == 2 || reward == 4);
            CHECK(euchre_env_step(env, EUCHRE_ACTION_PASS, &observation,
                                  legal, &reward) == EUCHRE_ENV_ERROR);
        }
    }
    euchre_env_destroy(env);
}

/* Require team control to expose only the currently acting partner's view. */
static void test_partnership_decision_environment(void) {
    EuchreEnv *env = euchre_env_create();
    CHECK(env != NULL);

    for (int team = 0; team < 2; ++team) {
        bool observed_partner[EUCHRE_PLAYERS] = {false};
        for (uint64_t seed = 1; seed <= 20; ++seed) {
            EuchreObservation observation;
            bool legal[EUCHRE_ACTION_COUNT];
            int reward;
            EuchreEnvStatus status = euchre_env_reset_team(
                env, seed, team, &observation, legal, &reward);
            CHECK(status != EUCHRE_ENV_ERROR);

            while (status == EUCHRE_ENV_DECISION) {
                CHECK(observation.player == observation.current_player);
                CHECK(observation.player % 2 == team);
                CHECK(count_mask(legal) > 1);
                observed_partner[observation.player] = true;
                status = euchre_env_step(
                    env, first_legal_action(legal), &observation, legal,
                    &reward);
                CHECK(status != EUCHRE_ENV_ERROR);
            }
            CHECK(status == EUCHRE_ENV_TERMINAL);
            CHECK(observation.player % 2 == team);
            CHECK(count_mask(legal) == 0);
        }
        CHECK(observed_partner[team]);
        CHECK(observed_partner[team + 2]);
        CHECK(!observed_partner[1 - team]);
        CHECK(!observed_partner[3 - team]);
    }
    euchre_env_destroy(env);
}

/* Confirm the foreign-language adapter returns only frozen numeric buffers. */
static void test_numeric_bridge(void) {
    CHECK(euchre_bridge_layout_version() == EUCHRE_OBSERVATION_LAYOUT_VERSION);
    CHECK(euchre_bridge_observation_size() == EUCHRE_OBSERVATION_SIZE);
    CHECK(euchre_bridge_action_count() == EUCHRE_ACTION_COUNT);

    EuchreEnv *env = euchre_bridge_create();
    CHECK(env != NULL);
    int16_t observation[EUCHRE_OBSERVATION_SIZE];
    uint8_t mask[EUCHRE_ENCODED_ACTION_MASK_SIZE];
    int reward;
    int status = euchre_bridge_reset(env, 1111, 1, observation, mask, &reward);
    CHECK(status == EUCHRE_ENV_DECISION);

    int decisions = 0;
    while (status == EUCHRE_ENV_DECISION) {
        CHECK(observation[EUCHRE_OBS_VERSION] ==
              EUCHRE_OBSERVATION_LAYOUT_VERSION);
        CHECK(observation[EUCHRE_OBS_PLAYER] == 1);
        int action = -1;
        int choices = 0;
        for (int candidate = 0; candidate < EUCHRE_ACTION_COUNT; ++candidate) {
            CHECK(mask[candidate] == 0 || mask[candidate] == 1);
            if (mask[candidate]) {
                if (action < 0) action = candidate;
                ++choices;
            }
        }
        CHECK(choices > 1);
        status = euchre_bridge_step(env, action, observation, mask, &reward);
        CHECK(status != EUCHRE_ENV_ERROR);
        CHECK(++decisions < MAX_HANDS_PER_MATCH);
    }
    CHECK(status == EUCHRE_ENV_TERMINAL);
    CHECK(observation[EUCHRE_OBS_PHASE] == EUCHRE_HAND_COMPLETE);
    euchre_bridge_destroy(env);
}

/* Confirm a rejected learning action does not consume the current decision. */
static void test_environment_rejects_masked_action(void) {
    EuchreEnv *env = euchre_env_create();
    CHECK(env != NULL);
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    int reward;
    CHECK(euchre_env_reset(env, 909, 1, &observation, legal, &reward) ==
          EUCHRE_ENV_DECISION);

    EuchreAction invalid = EUCHRE_ACTION_INVALID;
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        if (!legal[action]) {
            invalid = (EuchreAction)action;
            break;
        }
    }
    CHECK(invalid != EUCHRE_ACTION_INVALID);
    CHECK(euchre_env_step(env, invalid, &observation, legal, &reward) ==
          EUCHRE_ENV_ERROR);

    EuchreAction valid = first_legal_action(legal);
    CHECK(valid != EUCHRE_ACTION_INVALID);
    CHECK(euchre_env_step(env, valid, &observation, legal, &reward) !=
          EUCHRE_ENV_ERROR);
    euchre_env_destroy(env);
}

/* Count every owned or played card and require each deck ID exactly once. */
static bool cards_are_conserved(const EuchreGame *game) {
    int occurrences[EUCHRE_DECK_SIZE] = {0};
    size_t total = 0;

    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        for (size_t i = 0; i < game->hands[player].count; ++i) {
            int card_id = euchre_card_id(game->hands[player].cards[i]);
            if (card_id < 0) return false;
            ++occurrences[card_id];
            ++total;
        }
    }
    for (size_t i = 0; i < game->kitty_count; ++i) {
        int card_id = euchre_card_id(game->kitty[i]);
        if (card_id < 0) return false;
        ++occurrences[card_id];
        ++total;
    }
    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        if (game->cards_played[card_id]) {
            ++occurrences[card_id];
            ++total;
        }
    }
    if (total != EUCHRE_DECK_SIZE) return false;
    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        if (occurrences[card_id] != 1) return false;
    }
    return true;
}

/* Compare dealt locations to prove separate game objects shuffle independently. */
static bool deals_match(const EuchreGame *left, const EuchreGame *right) {
    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        for (size_t i = 0; i < EUCHRE_HAND_SIZE; ++i) {
            if (euchre_card_id(left->hands[player].cards[i]) !=
                euchre_card_id(right->hands[player].cards[i])) {
                return false;
            }
        }
    }
    for (size_t i = 0; i < EUCHRE_KITTY_SIZE; ++i) {
        if (euchre_card_id(left->kitty[i]) !=
            euchre_card_id(right->kitty[i])) {
            return false;
        }
    }
    return true;
}

/* Confirm every normal, euchred, march, and lone scoring outcome exactly. */
static void test_exact_scoring(void) {
    struct {
        int tricks;
        bool alone;
        int maker_points;
        int defender_points;
    } cases[] = {
        {0, false, 0, 2}, {2, false, 0, 2}, {3, false, 1, 0},
        {4, false, 1, 0}, {5, false, 2, 0}, {0, true, 0, 2},
        {2, true, 0, 2},  {3, true, 1, 0},  {4, true, 1, 0},
        {5, true, 4, 0},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int maker_points = -1;
        int defender_points = -1;
        CHECK(euchre_hand_points(cases[i].tricks, cases[i].alone,
                                 &maker_points, &defender_points));
        CHECK(maker_points == cases[i].maker_points);
        CHECK(defender_points == cases[i].defender_points);
    }

    int maker_points;
    int defender_points;
    CHECK(!euchre_hand_points(-1, false, &maker_points, &defender_points));
    CHECK(!euchre_hand_points(6, false, &maker_points, &defender_points));
}

/* Confirm interleaved games with the same seed still receive identical deals. */
static void test_per_game_random_state(void) {
    EuchreGame first;
    EuchreGame second;
    EuchreGame unrelated;
    euchre_init(&first, 5150);
    euchre_init(&second, 5150);
    euchre_init(&unrelated, 9999);

    euchre_deal(&first);
    euchre_deal(&unrelated);
    euchre_deal(&second);
    CHECK(deals_match(&first, &second));
}

/* Confirm all 24 cards remain unique through pickup, discard, and every play. */
static void test_card_conservation(void) {
    EuchreGame game;
    bool legal[EUCHRE_ACTION_COUNT];
    EuchreObservation observation;
    euchre_init(&game, 606);
    euchre_deal(&game);
    EuchreCard original_upcard = game.upcard;

    CHECK(game.kitty_count == EUCHRE_KITTY_SIZE);
    CHECK(cards_are_conserved(&game));
    CHECK(euchre_apply_action(&game, EUCHRE_ACTION_ORDER_UP));
    CHECK(game.bid_history_count == 1);
    CHECK(game.bid_history[0].action == EUCHRE_BID_ORDER_UP);
    CHECK(game.bid_history[0].round == 1);
    CHECK(game.bid_history[0].suit == game.turned_suit);
    CHECK(!game.bid_history[0].going_alone);
    CHECK(game.kitty_count == EUCHRE_KITTY_SIZE - 1);
    CHECK(cards_are_conserved(&game));

    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(euchre_apply_action(&game, first_card_action(legal)));
    CHECK(game.kitty_count == EUCHRE_KITTY_SIZE);
    CHECK(cards_are_conserved(&game));
    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(euchre_card_id(observation.upcard) == euchre_card_id(original_upcard));

    while (game.phase == EUCHRE_PLAYING) {
        CHECK(euchre_policy_legal_actions(&game, legal) > 0);
        CHECK(euchre_apply_action(&game, first_card_action(legal)));
        CHECK(cards_are_conserved(&game));
    }
}

/*
 * Prove hidden state cannot influence an observation while public actions do.
 * The copied game deliberately contains different opponent cards, kitty cards,
 * kitty size, and RNG state but must produce the same bytes for the same seat.
 */
static void test_observation_privacy_and_public_events(void) {
    EuchreGame game;
    bool legal[EUCHRE_ACTION_COUNT];
    euchre_init(&game, 808);
    euchre_deal(&game);

    CHECK(euchre_apply_action(&game, EUCHRE_ACTION_ORDER_UP));
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(euchre_apply_action(&game, first_card_action(legal)));

    /* Record one public play before comparing the two otherwise equal states. */
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE);
    CHECK(euchre_apply_action(&game, first_card_action(legal)));
    int observing_player = game.current_player;

    EuchreObservation baseline;
    CHECK(euchre_observe(&game, observing_player, &baseline));
    CHECK(baseline.bid_history_count == 1);
    CHECK(baseline.trick_history_used[0][game.leader]);

    EuchreGame hidden_changed = game;
    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        if (player == observing_player) continue;
        for (size_t i = 0; i < hidden_changed.hands[player].count; ++i) {
            int old_id = euchre_card_id(hidden_changed.hands[player].cards[i]);
            hidden_changed.hands[player].cards[i] =
                euchre_card_from_id((old_id + 7) % EUCHRE_DECK_SIZE);
        }
    }
    for (size_t i = 0; i < EUCHRE_KITTY_SIZE; ++i) {
        hidden_changed.kitty[i] = euchre_card_from_id((int)(i + 12));
    }
    hidden_changed.kitty_count = 1;
    hidden_changed.rng_state ^= UINT64_C(0xffffffffffffffff);

    EuchreObservation after_hidden_changes;
    CHECK(euchre_observe(&hidden_changed, observing_player,
                         &after_hidden_changes));
    CHECK(memcmp(&baseline, &after_hidden_changes, sizeof(baseline)) == 0);
    int16_t baseline_buffer[EUCHRE_OBSERVATION_SIZE];
    int16_t hidden_buffer[EUCHRE_OBSERVATION_SIZE];
    CHECK(euchre_encode_observation(&baseline, baseline_buffer));
    CHECK(euchre_encode_observation(&after_hidden_changes, hidden_buffer));
    CHECK(memcmp(baseline_buffer, hidden_buffer,
                 sizeof(baseline_buffer)) == 0);

    /* A real public play must immediately appear in that same seat's view. */
    CHECK(euchre_policy_legal_actions(&game, legal) > 0);
    EuchreAction public_play = first_card_action(legal);
    CHECK(euchre_apply_action(&game, public_play));

    EuchreObservation after_public_play;
    CHECK(euchre_observe(&game, observing_player, &after_public_play));
    CHECK(memcmp(&baseline, &after_public_play, sizeof(baseline)) != 0);
    CHECK(after_public_play.cards_played[public_play]);
    CHECK(after_public_play.trick_history_used[0][observing_player]);
    CHECK(euchre_card_id(
              after_public_play.trick_history[0][observing_player]) ==
          public_play);
}

/* Confirm every frozen numeric section, sentinel, and byte mask is encoded. */
static void test_numeric_buffer_layout(void) {
    EuchreGame game;
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    int16_t encoded[EUCHRE_OBSERVATION_SIZE];
    uint8_t encoded_mask[EUCHRE_ENCODED_ACTION_MASK_SIZE];
    euchre_init(&game, 1001);
    euchre_deal(&game);

    CHECK(euchre_apply_action(&game, EUCHRE_ACTION_ORDER_UP));
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE + 1);
    CHECK(euchre_apply_action(&game, first_card_action(legal)));
    int leader = game.current_player;
    CHECK(euchre_policy_legal_actions(&game, legal) == EUCHRE_HAND_SIZE);
    EuchreAction led_card = first_card_action(legal);
    CHECK(euchre_apply_action(&game, led_card));

    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(euchre_encode_observation(&observation, encoded));
    CHECK(encoded[EUCHRE_OBS_VERSION] == EUCHRE_OBSERVATION_LAYOUT_VERSION);
    CHECK(encoded[EUCHRE_OBS_PLAYER] == observation.player);
    CHECK(encoded[EUCHRE_OBS_DEALER] == observation.dealer);
    CHECK(encoded[EUCHRE_OBS_CURRENT_PLAYER] == observation.current_player);
    CHECK(encoded[EUCHRE_OBS_LEADER] == leader);
    CHECK(encoded[EUCHRE_OBS_CALLER] == observation.caller);
    CHECK(encoded[EUCHRE_OBS_PHASE] == EUCHRE_PLAYING);
    CHECK(encoded[EUCHRE_OBS_TRUMP] == (int16_t)observation.trump);
    CHECK(encoded[EUCHRE_OBS_UPCARD] == euchre_card_id(observation.upcard));
    CHECK(encoded[EUCHRE_OBS_BID_HISTORY_COUNT] == 1);
    CHECK(encoded[EUCHRE_OBS_TRICKS_PLAYED] == 0);
    CHECK(encoded[EUCHRE_OBS_TRICK_PLAYS] == 1);

    int hand_cards = 0;
    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        int expected_hand = 0;
        for (size_t i = 0; i < observation.hand_count; ++i) {
            if (euchre_card_id(observation.hand[i]) == card_id) {
                expected_hand = 1;
            }
        }
        CHECK(encoded[EUCHRE_OBS_HAND_MASK + card_id] == expected_hand);
        CHECK(encoded[EUCHRE_OBS_PLAYED_MASK + card_id] ==
              (observation.cards_played[card_id] ? 1 : 0));
        hand_cards += encoded[EUCHRE_OBS_HAND_MASK + card_id];
    }
    CHECK(hand_cards == (int)observation.hand_count);
    CHECK(encoded[EUCHRE_OBS_PLAYED_MASK + led_card] == 1);

    int bid_offset = EUCHRE_OBS_BID_HISTORY;
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_PLAYER] ==
          observation.bid_history[0].player);
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_ROUND] == 1);
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_ACTION] ==
          EUCHRE_BID_ORDER_UP);
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_SUIT] ==
          (int16_t)game.trump);
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_ALONE] == 0);
    CHECK(encoded[bid_offset + EUCHRE_ENCODED_BID_WIDTH] ==
          EUCHRE_ENCODED_NONE);

    CHECK(encoded[EUCHRE_OBS_TRICK_HISTORY + leader] == led_card);
    CHECK(encoded[EUCHRE_OBS_TRICK_LEADERS] == leader);
    CHECK(encoded[EUCHRE_OBS_TRICK_WINNERS] == EUCHRE_ENCODED_NONE);
    CHECK(encoded[EUCHRE_OBS_TRICK_HISTORY + EUCHRE_PLAYERS] ==
          EUCHRE_ENCODED_NONE);

    CHECK(euchre_policy_legal_actions(&game, legal) > 0);
    CHECK(euchre_encode_action_mask(legal, encoded_mask));
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        CHECK(encoded_mask[action] == (legal[action] ? 1 : 0));
    }
}

/* Confirm bidding events retain their actor, round, action, and selected suit. */
static void test_bidding_history(void) {
    EuchreGame game;
    EuchreObservation observation;
    bool legal[EUCHRE_ACTION_COUNT];
    euchre_init(&game, 707);
    euchre_deal(&game);
    int first_bidder = game.current_player;

    for (int i = 0; i < EUCHRE_PLAYERS; ++i) {
        CHECK(euchre_apply_action(&game, EUCHRE_ACTION_PASS));
        CHECK(game.bid_history[i].player == (first_bidder + i) % EUCHRE_PLAYERS);
        CHECK(game.bid_history[i].round == 1);
        CHECK(game.bid_history[i].action == EUCHRE_BID_PASS);
        CHECK(game.bid_history[i].suit == EUCHRE_NO_SUIT);
    }

    for (int i = 0; i < 2; ++i) {
        CHECK(euchre_apply_action(&game, EUCHRE_ACTION_PASS));
    }
    CHECK(euchre_policy_legal_actions(&game, legal) == 7);
    EuchreAction call = EUCHRE_ACTION_INVALID;
    for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
        EuchreAction candidate = (EuchreAction)(EUCHRE_ACTION_CALL_CLUBS + suit);
        if (legal[candidate]) {
            call = candidate;
            break;
        }
    }
    CHECK(call != EUCHRE_ACTION_INVALID);
    int caller = game.current_player;
    CHECK(euchre_apply_action(&game, call));
    CHECK(game.bid_history_count == 7);

    EuchreBidRecord final_bid = game.bid_history[6];
    CHECK(final_bid.player == caller);
    CHECK(final_bid.round == 2);
    CHECK(final_bid.action == EUCHRE_BID_CALL_TRUMP);
    CHECK(final_bid.suit == (EuchreSuit)(call - EUCHRE_ACTION_CALL_CLUBS));
    CHECK(!final_bid.going_alone);

    CHECK(euchre_observe(&game, game.current_player, &observation));
    CHECK(observation.bid_history_count == game.bid_history_count);
    for (size_t i = 0; i < observation.bid_history_count; ++i) {
        CHECK(observation.bid_history[i].player == game.bid_history[i].player);
        CHECK(observation.bid_history[i].round == game.bid_history[i].round);
        CHECK(observation.bid_history[i].action == game.bid_history[i].action);
        CHECK(observation.bid_history[i].suit == game.bid_history[i].suit);
    }
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
    CHECK(observation.trick_leaders[0] == leader);
    CHECK(observation.trick_history_used[0][leader]);
    CHECK(euchre_card_id(observation.trick_history[0][leader]) == led_action);

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
    for (int trick = 0; trick < EUCHRE_HAND_SIZE; ++trick) {
        CHECK(observation.trick_leaders[trick] >= 0);
        CHECK(observation.trick_winners[trick] >= 0);
        if (trick > 0) {
            CHECK(observation.trick_leaders[trick] ==
                  observation.trick_winners[trick - 1]);
        }
        int cards_in_trick = 0;
        for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
            if (observation.trick_history_used[trick][player]) ++cards_in_trick;
        }
        CHECK(cards_in_trick == EUCHRE_PLAYERS);
    }
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
    for (int trick = 0; trick < EUCHRE_HAND_SIZE; ++trick) {
        CHECK(!game.trick_history_used[trick][game.sitting_out]);
        int cards_in_trick = 0;
        for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
            if (game.trick_history_used[trick][player]) ++cards_in_trick;
        }
        CHECK(cards_in_trick == EUCHRE_PLAYERS - 1);
    }
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
    test_numeric_bridge();
    test_decision_point_environment();
    test_partnership_decision_environment();
    test_environment_rejects_masked_action();
    test_exact_scoring();
    test_per_game_random_state();
    test_card_conservation();
    test_observation_privacy_and_public_events();
    test_numeric_buffer_layout();
    test_bidding_history();
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

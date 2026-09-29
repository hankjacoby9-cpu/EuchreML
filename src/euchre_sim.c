#include "euchre_sim.h"

/* Advance the simulator's deterministic random generator. */
static uint32_t next_bot_random(uint64_t *state) {
    if (*state == 0) *state = UINT64_C(0xd1b54a32d192ed03);
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (uint32_t)(*state >> 16);
}

/* Estimate a bidding hand by counting cards that would act as the given suit. */
static int count_suit(const EuchreObservation *observation, EuchreSuit suit) {
    int count = 0;
    for (size_t i = 0; i < observation->hand_count; ++i) {
        if (euchre_effective_suit(observation->hand[i], suit) == suit) ++count;
    }
    return count;
}

/* Choose the first legal fixed card ID, used for the baseline dealer discard. */
static EuchreAction first_legal_card(const bool legal_actions[]) {
    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        if (legal_actions[card_id]) return (EuchreAction)card_id;
    }
    return EUCHRE_ACTION_INVALID;
}

/* Make one baseline decision using only public observation data and legal actions. */
EuchreAction euchre_simple_policy(
    const EuchreObservation *observation,
    const bool legal_actions[EUCHRE_ACTION_COUNT], void *context) {
    if (observation == NULL || legal_actions == NULL || context == NULL) {
        return EUCHRE_ACTION_INVALID;
    }

    if (observation->phase == EUCHRE_BIDDING_ROUND_ONE) {
        if (observation->trump != EUCHRE_NO_SUIT) {
            return first_legal_card(legal_actions);
        }
        int trump_count = count_suit(observation, observation->turned_suit);
        if (trump_count >= 2) {
            return trump_count >= 4 ? EUCHRE_ACTION_ORDER_UP_ALONE
                                    : EUCHRE_ACTION_ORDER_UP;
        }
        return EUCHRE_ACTION_PASS;
    }

    if (observation->phase == EUCHRE_BIDDING_ROUND_TWO) {
        EuchreSuit best_suit = EUCHRE_NO_SUIT;
        int best_count = -1;
        for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
            EuchreAction call = (EuchreAction)(EUCHRE_ACTION_CALL_CLUBS + suit);
            if (!legal_actions[call]) continue;
            int count = count_suit(observation, (EuchreSuit)suit);
            if (count > best_count) {
                best_count = count;
                best_suit = (EuchreSuit)suit;
            }
        }
        if (best_count >= 2 || !legal_actions[EUCHRE_ACTION_PASS]) {
            int first_call = best_count >= 4 ? EUCHRE_ACTION_CALL_CLUBS_ALONE
                                             : EUCHRE_ACTION_CALL_CLUBS;
            return (EuchreAction)(first_call + best_suit);
        }
        return EUCHRE_ACTION_PASS;
    }

    if (observation->phase == EUCHRE_PLAYING) {
        EuchreAction card_actions[EUCHRE_HAND_SIZE];
        size_t count = 0;
        for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
            if (legal_actions[card_id]) {
                card_actions[count++] = (EuchreAction)card_id;
            }
        }
        if (count == 0) return EUCHRE_ACTION_INVALID;
        uint64_t *bot_state = context;
        return card_actions[next_bot_random(bot_state) % count];
    }
    return EUCHRE_ACTION_INVALID;
}

/* Deal and play a hand with four copies of the baseline policy. */
bool euchre_play_simple_hand(EuchreGame *game, uint64_t *bot_state) {
    if (game == NULL || bot_state == NULL) return false;
    EuchrePolicy policies[EUCHRE_PLAYERS];
    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        policies[player] = (EuchrePolicy){"simple", euchre_simple_policy,
                                          bot_state};
    }
    return euchre_play_policy_hand(game, policies);
}

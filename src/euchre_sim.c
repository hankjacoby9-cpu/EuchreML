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
static int count_suit(const EuchreHand *hand, EuchreSuit suit) {
    int count = 0;
    for (size_t i = 0; i < hand->count; ++i) {
        if (euchre_effective_suit(hand->cards[i], suit) == suit) ++count;
    }
    return count;
}

/* Complete both bidding rounds with a deliberately simple suit-count strategy. */
static bool run_simple_bidding(EuchreGame *game) {
    while (game->phase == EUCHRE_BIDDING_ROUND_ONE) {
        int player = game->current_player;
        int trump_count = count_suit(&game->hands[player], game->turned_suit);
        if (trump_count >= 2) {
            if (!euchre_order_up(game, trump_count >= 4)) return false;
            return euchre_dealer_discard(game, 0);
        }
        if (!euchre_pass_bid(game)) return false;
    }

    while (game->phase == EUCHRE_BIDDING_ROUND_TWO) {
        int player = game->current_player;
        EuchreSuit best_suit = EUCHRE_NO_SUIT;
        int best_count = -1;

        /* Check each legal suit and remember the one represented most in the hand. */
        for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
            if ((EuchreSuit)suit == game->turned_suit) continue;
            int count = count_suit(&game->hands[player], (EuchreSuit)suit);
            if (count > best_count) {
                best_count = count;
                best_suit = (EuchreSuit)suit;
            }
        }

        if (best_count >= 2 || player == game->dealer) {
            return euchre_call_trump(game, best_suit, best_count >= 4);
        }
        if (!euchre_pass_bid(game)) return false;
    }
    return false;
}

/* Deal, bid, and play a complete hand while rejecting impossible engine states. */
bool euchre_play_simple_hand(EuchreGame *game, uint64_t *bot_state) {
    if (game == NULL || bot_state == NULL) return false;

    euchre_deal(game);
    if (!run_simple_bidding(game)) return false;

    while (game->phase == EUCHRE_PLAYING) {
        size_t legal[EUCHRE_HAND_SIZE];
        size_t legal_count = euchre_legal_moves(game, game->current_player, legal);
        if (legal_count == 0) return false;

        size_t choice = legal[next_bot_random(bot_state) % legal_count];
        if (!euchre_play_card(game, choice, NULL)) return false;
    }
    return game->phase == EUCHRE_HAND_COMPLETE;
}

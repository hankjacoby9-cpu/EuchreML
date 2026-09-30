#include "euchre_policy.h"

#include <string.h>

#define MAX_ACTIONS_PER_HAND 64

/* Find a permanent card ID in a hand whose storage order may change after plays. */
static bool find_card(const EuchreHand *hand, int card_id, size_t *hand_index) {
    for (size_t i = 0; i < hand->count; ++i) {
        if (euchre_card_id(hand->cards[i]) == card_id) {
            *hand_index = i;
            return true;
        }
    }
    return false;
}

/* Build a private-information-safe view for one seat. */
bool euchre_observe(const EuchreGame *game, int player,
                    EuchreObservation *observation) {
    if (game == NULL || observation == NULL || player < 0 ||
        player >= EUCHRE_PLAYERS) {
        return false;
    }

    memset(observation, 0, sizeof(*observation));
    observation->player = player;
    observation->hand_count = game->hands[player].count;
    memcpy(observation->hand, game->hands[player].cards,
           observation->hand_count * sizeof(EuchreCard));
    observation->upcard = game->upcard;
    memcpy(observation->trick, game->trick, sizeof(observation->trick));
    memcpy(observation->trick_slot_used, game->trick_slot_used,
           sizeof(observation->trick_slot_used));
    memcpy(observation->cards_played, game->cards_played,
           sizeof(observation->cards_played));
    observation->dealer = game->dealer;
    observation->current_player = game->current_player;
    observation->leader = game->leader;
    observation->bid_turns = game->bid_turns;
    observation->tricks_played = game->tricks_played;
    observation->tricks_won[0] = game->tricks_won[0];
    observation->tricks_won[1] = game->tricks_won[1];
    observation->score[0] = game->score[0];
    observation->score[1] = game->score[1];
    observation->caller = game->caller;
    observation->sitting_out = game->sitting_out;
    observation->going_alone = game->going_alone;
    observation->trump = game->trump;
    observation->turned_suit = game->turned_suit;
    observation->phase = game->phase;
    return true;
}

/* Add all legal fixed card IDs for a discard or play decision. */
static size_t add_card_actions(const EuchreGame *game, bool legal_actions[],
                               bool is_discard) {
    const EuchreHand *hand = &game->hands[game->current_player];
    size_t count = 0;
    for (size_t i = 0; i < hand->count; ++i) {
        if (!is_discard &&
            !euchre_is_legal_play(game, game->current_player, i)) {
            continue;
        }
        int card_id = euchre_card_id(hand->cards[i]);
        legal_actions[card_id] = true;
        ++count;
    }
    return count;
}

/* Build the fixed-size legal mask consumed by every policy implementation. */
size_t euchre_policy_legal_actions(
    const EuchreGame *game, bool legal_actions[EUCHRE_ACTION_COUNT]) {
    if (game == NULL || legal_actions == NULL) return 0;
    memset(legal_actions, 0, sizeof(bool) * EUCHRE_ACTION_COUNT);

    if (game->phase == EUCHRE_BIDDING_ROUND_ONE) {
        if (game->trump != EUCHRE_NO_SUIT) {
            return add_card_actions(game, legal_actions, true);
        }
        legal_actions[EUCHRE_ACTION_PASS] = true;
        legal_actions[EUCHRE_ACTION_ORDER_UP] = true;
        legal_actions[EUCHRE_ACTION_ORDER_UP_ALONE] = true;
        return 3;
    }

    if (game->phase == EUCHRE_BIDDING_ROUND_TWO) {
        size_t count = 0;
        bool dealer_is_stuck = game->current_player == game->dealer &&
                               game->bid_turns == EUCHRE_PLAYERS - 1;
        if (!dealer_is_stuck) {
            legal_actions[EUCHRE_ACTION_PASS] = true;
            ++count;
        }
        for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
            if ((EuchreSuit)suit == game->turned_suit) continue;
            legal_actions[EUCHRE_ACTION_CALL_CLUBS + suit] = true;
            legal_actions[EUCHRE_ACTION_CALL_CLUBS_ALONE + suit] = true;
            count += 2;
        }
        return count;
    }

    if (game->phase == EUCHRE_PLAYING) {
        return add_card_actions(game, legal_actions, false);
    }
    return 0;
}

/* Validate and apply one fixed action to the current game state. */
bool euchre_apply_action(EuchreGame *game, EuchreAction action) {
    bool legal_actions[EUCHRE_ACTION_COUNT];
    if (game == NULL || action < 0 || action >= EUCHRE_ACTION_COUNT ||
        euchre_policy_legal_actions(game, legal_actions) == 0 ||
        !legal_actions[action]) {
        return false;
    }

    if (action < EUCHRE_DECK_SIZE) {
        size_t hand_index;
        if (!find_card(&game->hands[game->current_player], action, &hand_index)) {
            return false;
        }
        if (game->phase == EUCHRE_BIDDING_ROUND_ONE) {
            return euchre_dealer_discard(game, hand_index);
        }
        return euchre_play_card(game, hand_index, NULL);
    }

    if (action == EUCHRE_ACTION_PASS) return euchre_pass_bid(game);
    if (action == EUCHRE_ACTION_ORDER_UP) return euchre_order_up(game, false);
    if (action == EUCHRE_ACTION_ORDER_UP_ALONE) return euchre_order_up(game, true);

    bool alone = action >= EUCHRE_ACTION_CALL_CLUBS_ALONE;
    int first_call = alone ? EUCHRE_ACTION_CALL_CLUBS_ALONE
                           : EUCHRE_ACTION_CALL_CLUBS;
    EuchreSuit suit = (EuchreSuit)(action - first_call);
    return euchre_call_trump(game, suit, alone);
}

/* Ask the correct seat's policy for actions until one complete hand finishes. */
bool euchre_play_policy_hand(EuchreGame *game,
                             const EuchrePolicy policies[EUCHRE_PLAYERS]) {
    if (game == NULL || policies == NULL) return false;
    euchre_deal(game);

    for (int decision = 0; decision < MAX_ACTIONS_PER_HAND; ++decision) {
        if (game->phase == EUCHRE_HAND_COMPLETE) return true;
        int player = game->current_player;
        if (player < 0 || player >= EUCHRE_PLAYERS ||
            policies[player].choose_action == NULL) {
            return false;
        }

        EuchreObservation observation;
        bool legal_actions[EUCHRE_ACTION_COUNT];
        if (!euchre_observe(game, player, &observation) ||
            euchre_policy_legal_actions(game, legal_actions) == 0) {
            return false;
        }
        EuchreAction action = policies[player].choose_action(
            &observation, legal_actions, policies[player].context);
        if (!euchre_apply_action(game, action)) return false;
    }
    return false;
}

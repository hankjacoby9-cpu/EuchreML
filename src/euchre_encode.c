#include "euchre_encode.h"

#include <string.h>

/* Guard the numeric action contract against accidental enum reordering. */
_Static_assert(EUCHRE_ACTION_PASS == 24, "pass action ID changed");
_Static_assert(EUCHRE_ACTION_ORDER_UP == 25, "order-up action ID changed");
_Static_assert(EUCHRE_ACTION_ORDER_UP_ALONE == 26,
               "alone order-up action ID changed");
_Static_assert(EUCHRE_ACTION_CALL_CLUBS == 27, "first call action ID changed");
_Static_assert(EUCHRE_ACTION_CALL_SPADES_ALONE == 34,
               "last call action ID changed");
_Static_assert(EUCHRE_ACTION_COUNT == 35, "action count changed");
_Static_assert(EUCHRE_OBS_TRICK_WINNERS + EUCHRE_HAND_SIZE ==
                   EUCHRE_OBSERVATION_SIZE,
               "observation offsets do not match buffer size");

/* Encode a public observation into the frozen version-1 integer layout. */
bool euchre_encode_observation(
    const EuchreObservation *observation,
    int16_t output[EUCHRE_OBSERVATION_SIZE]) {
    if (observation == NULL || output == NULL ||
        observation->hand_count > EUCHRE_HAND_CAPACITY ||
        observation->bid_history_count > EUCHRE_MAX_BIDS) {
        return false;
    }

    for (int i = 0; i < EUCHRE_OBSERVATION_SIZE; ++i) {
        output[i] = EUCHRE_ENCODED_NONE;
    }

    output[EUCHRE_OBS_VERSION] = EUCHRE_OBSERVATION_LAYOUT_VERSION;
    output[EUCHRE_OBS_PLAYER] = (int16_t)observation->player;
    output[EUCHRE_OBS_DEALER] = (int16_t)observation->dealer;
    output[EUCHRE_OBS_CURRENT_PLAYER] =
        (int16_t)observation->current_player;
    output[EUCHRE_OBS_LEADER] = (int16_t)observation->leader;
    output[EUCHRE_OBS_CALLER] = (int16_t)observation->caller;
    output[EUCHRE_OBS_SITTING_OUT] = (int16_t)observation->sitting_out;
    output[EUCHRE_OBS_GOING_ALONE] = observation->going_alone ? 1 : 0;
    output[EUCHRE_OBS_PHASE] = (int16_t)observation->phase;
    output[EUCHRE_OBS_TRUMP] = (int16_t)observation->trump;
    output[EUCHRE_OBS_TURNED_SUIT] = (int16_t)observation->turned_suit;
    output[EUCHRE_OBS_UPCARD] = (int16_t)euchre_card_id(observation->upcard);
    output[EUCHRE_OBS_BID_TURNS] = (int16_t)observation->bid_turns;
    output[EUCHRE_OBS_BID_HISTORY_COUNT] =
        (int16_t)observation->bid_history_count;
    output[EUCHRE_OBS_TRICKS_PLAYED] = (int16_t)observation->tricks_played;
    output[EUCHRE_OBS_TRICK_PLAYS] = (int16_t)observation->trick_plays;
    output[EUCHRE_OBS_TEAM_0_TRICKS] = (int16_t)observation->tricks_won[0];
    output[EUCHRE_OBS_TEAM_1_TRICKS] = (int16_t)observation->tricks_won[1];
    output[EUCHRE_OBS_TEAM_0_SCORE] = (int16_t)observation->score[0];
    output[EUCHRE_OBS_TEAM_1_SCORE] = (int16_t)observation->score[1];

    memset(&output[EUCHRE_OBS_HAND_MASK], 0,
           sizeof(int16_t) * EUCHRE_DECK_SIZE);
    for (size_t i = 0; i < observation->hand_count; ++i) {
        int card_id = euchre_card_id(observation->hand[i]);
        if (card_id < 0) return false;
        output[EUCHRE_OBS_HAND_MASK + card_id] = 1;
    }

    for (int card_id = 0; card_id < EUCHRE_DECK_SIZE; ++card_id) {
        output[EUCHRE_OBS_PLAYED_MASK + card_id] =
            observation->cards_played[card_id] ? 1 : 0;
    }

    for (size_t bid = 0; bid < observation->bid_history_count; ++bid) {
        int offset = EUCHRE_OBS_BID_HISTORY +
                     (int)bid * EUCHRE_ENCODED_BID_WIDTH;
        const EuchreBidRecord *record = &observation->bid_history[bid];
        output[offset + EUCHRE_ENCODED_BID_PLAYER] = (int16_t)record->player;
        output[offset + EUCHRE_ENCODED_BID_ROUND] = (int16_t)record->round;
        output[offset + EUCHRE_ENCODED_BID_ACTION] = (int16_t)record->action;
        output[offset + EUCHRE_ENCODED_BID_SUIT] = (int16_t)record->suit;
        output[offset + EUCHRE_ENCODED_BID_ALONE] =
            record->going_alone ? 1 : 0;
    }

    for (int trick = 0; trick < EUCHRE_HAND_SIZE; ++trick) {
        for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
            int offset = EUCHRE_OBS_TRICK_HISTORY +
                         trick * EUCHRE_PLAYERS + player;
            if (observation->trick_history_used[trick][player]) {
                int card_id =
                    euchre_card_id(observation->trick_history[trick][player]);
                if (card_id < 0) return false;
                output[offset] = (int16_t)card_id;
            }
        }
        output[EUCHRE_OBS_TRICK_LEADERS + trick] =
            (int16_t)observation->trick_leaders[trick];
        output[EUCHRE_OBS_TRICK_WINNERS + trick] =
            (int16_t)observation->trick_winners[trick];
    }
    return true;
}

/* Encode the legal-action mask as exact 0/1 bytes for foreign consumers. */
bool euchre_encode_action_mask(
    const bool legal_actions[EUCHRE_ACTION_COUNT],
    uint8_t output[EUCHRE_ENCODED_ACTION_MASK_SIZE]) {
    if (legal_actions == NULL || output == NULL) return false;
    for (int action = 0; action < EUCHRE_ACTION_COUNT; ++action) {
        output[action] = legal_actions[action] ? UINT8_C(1) : UINT8_C(0);
    }
    return true;
}

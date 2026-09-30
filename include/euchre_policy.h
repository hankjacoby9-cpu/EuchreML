#ifndef EUCHRE_POLICY_H
#define EUCHRE_POLICY_H

#include "euchre.h"

#include <stdbool.h>
#include <stddef.h>

/* Card actions use stable deck IDs 0-23 for both discarding and playing. */
typedef enum {
    EUCHRE_ACTION_CARD_0 = 0,
    EUCHRE_ACTION_PASS = EUCHRE_DECK_SIZE,
    EUCHRE_ACTION_ORDER_UP,
    EUCHRE_ACTION_ORDER_UP_ALONE,
    EUCHRE_ACTION_CALL_CLUBS,
    EUCHRE_ACTION_CALL_DIAMONDS,
    EUCHRE_ACTION_CALL_HEARTS,
    EUCHRE_ACTION_CALL_SPADES,
    EUCHRE_ACTION_CALL_CLUBS_ALONE,
    EUCHRE_ACTION_CALL_DIAMONDS_ALONE,
    EUCHRE_ACTION_CALL_HEARTS_ALONE,
    EUCHRE_ACTION_CALL_SPADES_ALONE,
    EUCHRE_ACTION_COUNT,
    EUCHRE_ACTION_INVALID = -1
} EuchreAction;

/* Everything a real player may observe, with private opponent cards omitted. */
typedef struct {
    EuchreCard hand[EUCHRE_HAND_CAPACITY];
    size_t hand_count;
    EuchreCard upcard;
    EuchreCard trick[EUCHRE_PLAYERS];
    bool trick_slot_used[EUCHRE_PLAYERS];
    EuchreBidRecord bid_history[EUCHRE_MAX_BIDS];
    size_t bid_history_count;
    EuchreCard trick_history[EUCHRE_HAND_SIZE][EUCHRE_PLAYERS];
    bool trick_history_used[EUCHRE_HAND_SIZE][EUCHRE_PLAYERS];
    int trick_leaders[EUCHRE_HAND_SIZE];
    int trick_winners[EUCHRE_HAND_SIZE];
    bool cards_played[EUCHRE_DECK_SIZE];
    int player;
    int dealer;
    int current_player;
    int leader;
    int bid_turns;
    int tricks_played;
    int tricks_won[2];
    int score[2];
    int caller;
    int sitting_out;
    bool going_alone;
    EuchreSuit trump;
    EuchreSuit turned_suit;
    EuchrePhase phase;
} EuchreObservation;

typedef EuchreAction (*EuchreChooseAction)(
    const EuchreObservation *observation,
    const bool legal_actions[EUCHRE_ACTION_COUNT], void *context);

typedef struct {
    const char *name;
    EuchreChooseAction choose_action;
    void *context;
} EuchrePolicy;

/* Build a private-information-safe view for one seat. */
bool euchre_observe(const EuchreGame *game, int player,
                    EuchreObservation *observation);

/* Build the fixed-size legal mask consumed by every policy implementation. */
size_t euchre_policy_legal_actions(
    const EuchreGame *game, bool legal_actions[EUCHRE_ACTION_COUNT]);

/* Validate and apply one fixed action to the current game state. */
bool euchre_apply_action(EuchreGame *game, EuchreAction action);

/* Ask the correct seat's policy for actions until one complete hand finishes. */
bool euchre_play_policy_hand(EuchreGame *game,
                             const EuchrePolicy policies[EUCHRE_PLAYERS]);

#endif

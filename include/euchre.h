#ifndef EUCHRE_H
#define EUCHRE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EUCHRE_PLAYERS 4
#define EUCHRE_HAND_SIZE 5
#define EUCHRE_HAND_CAPACITY 6
#define EUCHRE_DECK_SIZE 24
#define EUCHRE_KITTY_SIZE 4
#define EUCHRE_MAX_BIDS 8

typedef enum {
    EUCHRE_CLUBS,
    EUCHRE_DIAMONDS,
    EUCHRE_HEARTS,
    EUCHRE_SPADES,
    EUCHRE_NO_SUIT
} EuchreSuit;

typedef enum {
    EUCHRE_NINE,
    EUCHRE_TEN,
    EUCHRE_JACK,
    EUCHRE_QUEEN,
    EUCHRE_KING,
    EUCHRE_ACE
} EuchreRank;

typedef struct {
    EuchreSuit suit;
    EuchreRank rank;
} EuchreCard;

typedef struct {
    /* The dealer briefly has six cards after picking up the upcard. */
    EuchreCard cards[EUCHRE_HAND_CAPACITY];
    size_t count;
} EuchreHand;

typedef enum {
    EUCHRE_BIDDING_ROUND_ONE,
    EUCHRE_BIDDING_ROUND_TWO,
    EUCHRE_PLAYING,
    EUCHRE_HAND_COMPLETE
} EuchrePhase;

typedef enum {
    EUCHRE_BID_PASS,
    EUCHRE_BID_ORDER_UP,
    EUCHRE_BID_CALL_TRUMP
} EuchreBidAction;

/* One public bidding event, retained in the order in which it occurred. */
typedef struct {
    int player;
    int round;
    EuchreBidAction action;
    EuchreSuit suit;
    bool going_alone;
} EuchreBidRecord;

typedef struct {
    EuchreHand hands[EUCHRE_PLAYERS];
    EuchreCard kitty[EUCHRE_KITTY_SIZE];
    size_t kitty_count;
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
    int dealer;
    int current_player;
    int leader;
    int bid_turns;
    int trick_plays;
    int tricks_played;
    int tricks_won[2];
    int makers_team;
    int caller;
    int sitting_out;
    int score[2];
    bool going_alone;
    EuchreSuit trump;
    EuchreSuit turned_suit;
    EuchrePhase phase;
    uint64_t rng_state;
} EuchreGame;

/* Prepare a game and its random generator; a zero seed selects a default seed. */
void euchre_init(EuchreGame *game, uint64_t seed);
/* Rotate the dealer and deal a new hand, preserving the match score. */
void euchre_deal(EuchreGame *game);

/* First-round bidding action. Ordering up leaves the dealer with six cards. */
bool euchre_order_up(EuchreGame *game, bool go_alone);
bool euchre_pass_bid(EuchreGame *game);
/* Second-round bidding action; the turned-down suit cannot be selected. */
bool euchre_call_trump(EuchreGame *game, EuchreSuit suit, bool go_alone);
/* Return the dealer to five cards after an order-up and begin trick play. */
bool euchre_dealer_discard(EuchreGame *game, size_t hand_index);

/* Resolve the left bower's suit, validate actions, and expose legal choices. */
EuchreSuit euchre_effective_suit(EuchreCard card, EuchreSuit trump);
/* Convert cards to stable IDs 0-23 and back for policy actions and observations. */
int euchre_card_id(EuchreCard card);
EuchreCard euchre_card_from_id(int card_id);
bool euchre_is_legal_play(const EuchreGame *game, int player, size_t hand_index);
size_t euchre_legal_moves(const EuchreGame *game, int player,
                          size_t indices[EUCHRE_HAND_SIZE]);
/* Apply one play; trick_winner stays -1 until every active player has played. */
bool euchre_play_card(EuchreGame *game, size_t hand_index, int *trick_winner);

/* Calculate the exact points awarded for a completed hand. */
bool euchre_hand_points(int maker_tricks, bool going_alone, int *maker_points,
                        int *defender_points);

/* Helpers for displaying enums and cards without exposing formatting details. */
const char *euchre_suit_name(EuchreSuit suit);
const char *euchre_rank_name(EuchreRank rank);
void euchre_card_string(EuchreCard card, char *buffer, size_t buffer_size);

#endif

#include "euchre.h"

#include <stdio.h>
#include <string.h>

static uint64_t rng_state;

/* Advance the small deterministic random-number generator used when shuffling. */
static uint32_t next_random(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

/* Return the other suit with the same color, which identifies the left bower. */
static EuchreSuit same_color_suit(EuchreSuit suit) {
    switch (suit) {
        case EUCHRE_CLUBS: return EUCHRE_SPADES;
        case EUCHRE_SPADES: return EUCHRE_CLUBS;
        case EUCHRE_DIAMONDS: return EUCHRE_HEARTS;
        case EUCHRE_HEARTS: return EUCHRE_DIAMONDS;
        default: return EUCHRE_NO_SUIT;
    }
}

/* Move clockwise to the next seat, wrapping from player 3 back to player 0. */
static int next_player(int player) {
    return (player + 1) % EUCHRE_PLAYERS;
}

/* Reset trick state and give the opening lead to the player left of the dealer. */
static void begin_play(EuchreGame *game) {
    game->phase = EUCHRE_PLAYING;
    game->leader = next_player(game->dealer);
    game->current_player = game->leader;
    game->trick_plays = 0;
    game->tricks_played = 0;
    game->tricks_won[0] = 0;
    game->tricks_won[1] = 0;
    memset(game->trick_slot_used, 0, sizeof(game->trick_slot_used));
}

/* Initialize persistent game state and seed repeatable shuffles for simulations. */
void euchre_init(EuchreGame *game, uint64_t seed) {
    memset(game, 0, sizeof(*game));
    game->dealer = EUCHRE_PLAYERS - 1;
    game->trump = EUCHRE_NO_SUIT;
    game->turned_suit = EUCHRE_NO_SUIT;
    game->phase = EUCHRE_HAND_COMPLETE;
    rng_state = seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : seed;
}

/* Build and shuffle the 24-card deck, deal four hands, and reveal the upcard. */
void euchre_deal(EuchreGame *game) {
    EuchreCard deck[EUCHRE_DECK_SIZE];
    size_t deck_index = 0;

    game->dealer = next_player(game->dealer);
    /*
     * The suit and rank enums use consecutive integer values. Iterating over
     * them creates every combination from 9 through Ace in all four suits.
     */
    for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
        for (int rank = EUCHRE_NINE; rank <= EUCHRE_ACE; ++rank) {
            deck[deck_index++] = (EuchreCard){(EuchreSuit)suit, (EuchreRank)rank};
        }
    }

    for (size_t i = EUCHRE_DECK_SIZE - 1; i > 0; --i) {
        size_t j = next_random() % (i + 1);
        EuchreCard temporary = deck[i];
        deck[i] = deck[j];
        deck[j] = temporary;
    }

    deck_index = 0;
    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        game->hands[player].count = EUCHRE_HAND_SIZE;
        for (size_t card = 0; card < EUCHRE_HAND_SIZE; ++card) {
            game->hands[player].cards[card] = deck[deck_index++];
        }
    }
    for (size_t card = 0; card < EUCHRE_KITTY_SIZE; ++card) {
        game->kitty[card] = deck[deck_index++];
    }

    game->turned_suit = game->kitty[0].suit;
    game->trump = EUCHRE_NO_SUIT;
    game->caller = -1;
    game->makers_team = -1;
    game->bid_turns = 0;
    game->current_player = next_player(game->dealer);
    game->phase = EUCHRE_BIDDING_ROUND_ONE;
}

/* Accept a first-round bid and require the dealer to pick up the turned card. */
bool euchre_order_up(EuchreGame *game) {
    if (game->phase != EUCHRE_BIDDING_ROUND_ONE) return false;
    game->caller = game->current_player;
    game->makers_team = game->caller % 2;
    game->trump = game->turned_suit;

    EuchreHand *dealer_hand = &game->hands[game->dealer];
    dealer_hand->cards[dealer_hand->count++] = game->kitty[0];
    game->current_player = game->dealer;
    return true;
}

/* Remove one of the dealer's six cards, completing pickup and starting play. */
bool euchre_dealer_discard(EuchreGame *game, size_t hand_index) {
    EuchreHand *hand = &game->hands[game->dealer];
    if (game->phase != EUCHRE_BIDDING_ROUND_ONE || game->trump == EUCHRE_NO_SUIT ||
        game->current_player != game->dealer || hand->count != EUCHRE_HAND_SIZE + 1 ||
        hand_index >= hand->count) {
        return false;
    }
    hand->cards[hand_index] = hand->cards[hand->count - 1];
    --hand->count;
    begin_play(game);
    return true;
}

/* Record a pass, advance the bidder, and move to round two after four passes. */
bool euchre_pass_bid(EuchreGame *game) {
    if (game->phase != EUCHRE_BIDDING_ROUND_ONE &&
        game->phase != EUCHRE_BIDDING_ROUND_TWO) {
        return false;
    }
    if (game->phase == EUCHRE_BIDDING_ROUND_TWO &&
        game->current_player == game->dealer && game->bid_turns == 3) {
        /* Stick-the-dealer: after three second-round passes, passing is illegal. */
        return false;
    }

    ++game->bid_turns;
    if (game->bid_turns == EUCHRE_PLAYERS) {
        if (game->phase == EUCHRE_BIDDING_ROUND_ONE) {
            game->phase = EUCHRE_BIDDING_ROUND_TWO;
            game->bid_turns = 0;
            game->current_player = next_player(game->dealer);
        }
    } else {
        game->current_player = next_player(game->current_player);
    }
    return true;
}

/* Accept a legal second-round trump choice and start the five tricks. */
bool euchre_call_trump(EuchreGame *game, EuchreSuit suit) {
    if (game->phase != EUCHRE_BIDDING_ROUND_TWO || suit < EUCHRE_CLUBS ||
        suit > EUCHRE_SPADES || suit == game->turned_suit) {
        return false;
    }
    game->caller = game->current_player;
    game->makers_team = game->caller % 2;
    game->trump = suit;
    begin_play(game);
    return true;
}

/* Treat the left bower as trump when deciding which suit a card follows. */
EuchreSuit euchre_effective_suit(EuchreCard card, EuchreSuit trump) {
    if (card.rank == EUCHRE_JACK && card.suit == same_color_suit(trump)) {
        return trump;
    }
    return card.suit;
}

/* Assign a comparable value to a card for the current lead and trump suit. */
static int card_strength(EuchreCard card, EuchreSuit led, EuchreSuit trump) {
    if (card.rank == EUCHRE_JACK && card.suit == trump) return 200;
    if (card.rank == EUCHRE_JACK && card.suit == same_color_suit(trump)) return 199;

    EuchreSuit effective = euchre_effective_suit(card, trump);
    int rank_strength = (int)card.rank;
    if (effective == trump) return 100 + rank_strength;
    if (effective == led) return 50 + rank_strength;
    return rank_strength;
}

/* Check turn order, card position, and the requirement to follow the led suit. */
bool euchre_is_legal_play(const EuchreGame *game, int player, size_t hand_index) {
    if (game->phase != EUCHRE_PLAYING || player != game->current_player ||
        player < 0 || player >= EUCHRE_PLAYERS ||
        hand_index >= game->hands[player].count) {
        return false;
    }
    if (game->trick_plays == 0) return true;

    EuchreSuit led = euchre_effective_suit(game->trick[game->leader], game->trump);
    EuchreSuit chosen = euchre_effective_suit(game->hands[player].cards[hand_index],
                                               game->trump);
    if (chosen == led) return true;

    for (size_t i = 0; i < game->hands[player].count; ++i) {
        if (euchre_effective_suit(game->hands[player].cards[i], game->trump) == led) {
            return false;
        }
    }
    return true;
}

/* Collect the hand indices that the current player may legally choose. */
size_t euchre_legal_moves(const EuchreGame *game, int player,
                          size_t indices[EUCHRE_HAND_SIZE]) {
    size_t count = 0;
    if (player < 0 || player >= EUCHRE_PLAYERS) return 0;
    for (size_t i = 0; i < game->hands[player].count; ++i) {
        if (euchre_is_legal_play(game, player, i)) indices[count++] = i;
    }
    return count;
}

/* Compare all four played cards and return the seat that won the trick. */
static int resolve_trick(const EuchreGame *game) {
    EuchreSuit led = euchre_effective_suit(game->trick[game->leader], game->trump);
    int winner = game->leader;
    int best = card_strength(game->trick[winner], led, game->trump);
    for (int player = 0; player < EUCHRE_PLAYERS; ++player) {
        if (!game->trick_slot_used[player]) continue;
        int strength = card_strength(game->trick[player], led, game->trump);
        if (strength > best) {
            best = strength;
            winner = player;
        }
    }
    return winner;
}

/* Award points after five tricks based on whether the makers made their bid. */
static void score_hand(EuchreGame *game) {
    int maker_tricks = game->tricks_won[game->makers_team];
    if (maker_tricks == 5) {
        game->score[game->makers_team] += 2;
    } else if (maker_tricks >= 3) {
        game->score[game->makers_team] += 1;
    } else {
        game->score[1 - game->makers_team] += 2;
    }
    game->phase = EUCHRE_HAND_COMPLETE;
}

/* Play one legal card, then resolve the trick or finish and score the hand. */
bool euchre_play_card(EuchreGame *game, size_t hand_index, int *trick_winner) {
    int player = game->current_player;
    if (!euchre_is_legal_play(game, player, hand_index)) return false;

    EuchreHand *hand = &game->hands[player];
    game->trick[player] = hand->cards[hand_index];
    game->trick_slot_used[player] = true;
    hand->cards[hand_index] = hand->cards[hand->count - 1];
    --hand->count;
    ++game->trick_plays;

    if (game->trick_plays < EUCHRE_PLAYERS) {
        game->current_player = next_player(player);
        if (trick_winner != NULL) *trick_winner = -1;
        return true;
    }

    int winner = resolve_trick(game);
    ++game->tricks_won[winner % 2];
    ++game->tricks_played;
    if (trick_winner != NULL) *trick_winner = winner;

    if (game->tricks_played == EUCHRE_HAND_SIZE) {
        score_hand(game);
    } else {
        game->leader = winner;
        game->current_player = winner;
        game->trick_plays = 0;
        memset(game->trick_slot_used, 0, sizeof(game->trick_slot_used));
    }
    return true;
}

/* Convert a suit enum into text suitable for the command-line interface. */
const char *euchre_suit_name(EuchreSuit suit) {
    static const char *names[] = {"Clubs", "Diamonds", "Hearts", "Spades", "None"};
    return suit >= EUCHRE_CLUBS && suit <= EUCHRE_NO_SUIT ? names[suit] : "Invalid";
}

/* Convert a rank enum into text suitable for the command-line interface. */
const char *euchre_rank_name(EuchreRank rank) {
    static const char *names[] = {"9", "10", "Jack", "Queen", "King", "Ace"};
    return rank >= EUCHRE_NINE && rank <= EUCHRE_ACE ? names[rank] : "Invalid";
}

/* Format a complete card name, such as "Jack of Hearts," into a caller buffer. */
void euchre_card_string(EuchreCard card, char *buffer, size_t buffer_size) {
    if (buffer == NULL || buffer_size == 0) return;
    snprintf(buffer, buffer_size, "%s of %s", euchre_rank_name(card.rank),
             euchre_suit_name(card.suit));
}

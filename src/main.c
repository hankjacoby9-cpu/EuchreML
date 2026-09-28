#include "euchre.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Estimate hand strength for bidding by counting cards that act as a suit. */
static int count_suit(const EuchreHand *hand, EuchreSuit suit) {
    int count = 0;
    for (size_t i = 0; i < hand->count; ++i) {
        if (euchre_effective_suit(hand->cards[i], suit) == suit) ++count;
    }
    return count;
}

/* Drive both bidding rounds with a deliberately simple suit-count strategy. */
static void run_simple_bidding(EuchreGame *game) {
    while (game->phase == EUCHRE_BIDDING_ROUND_ONE) {
        int player = game->current_player;
        if (count_suit(&game->hands[player], game->turned_suit) >= 2) {
            euchre_order_up(game);
            euchre_dealer_discard(game, 0);
            return;
        }
        euchre_pass_bid(game);
    }

    while (game->phase == EUCHRE_BIDDING_ROUND_TWO) {
        int player = game->current_player;
        EuchreSuit best_suit = EUCHRE_NO_SUIT;
        int best_count = -1;
        for (int suit = EUCHRE_CLUBS; suit <= EUCHRE_SPADES; ++suit) {
            if ((EuchreSuit)suit == game->turned_suit) continue;
            int count = count_suit(&game->hands[player], (EuchreSuit)suit);
            if (count > best_count) {
                best_count = count;
                best_suit = (EuchreSuit)suit;
            }
        }
        if (best_count >= 2 || player == game->dealer) {
            euchre_call_trump(game, best_suit);
            return;
        }
        euchre_pass_bid(game);
    }
}

/* Deal and print one complete demonstration hand using simple automated players. */
int main(void) {
    EuchreGame game;
    euchre_init(&game, (uint64_t)time(NULL));
    euchre_deal(&game);

    char upcard[32];
    euchre_card_string(game.kitty[0], upcard, sizeof(upcard));
    printf("Dealer: Player %d | Upcard: %s\n", game.dealer, upcard);

    run_simple_bidding(&game);
    printf("Player %d called %s for Team %d\n\n", game.caller,
           euchre_suit_name(game.trump), game.makers_team);

    while (game.phase == EUCHRE_PLAYING) {
        int player = game.current_player;
        size_t legal[EUCHRE_HAND_SIZE];
        size_t legal_count = euchre_legal_moves(&game, player, legal);
        size_t choice = legal[rand() % legal_count];
        EuchreCard played = game.hands[player].cards[choice];
        int winner = -1;
        euchre_play_card(&game, choice, &winner);

        char card_name[32];
        euchre_card_string(played, card_name, sizeof(card_name));
        printf("Player %d plays %s\n", player, card_name);
        if (winner >= 0) printf("Player %d wins the trick\n\n", winner);
    }

    printf("Tricks: Team 0 = %d, Team 1 = %d\n", game.tricks_won[0],
           game.tricks_won[1]);
    printf("Score:  Team 0 = %d, Team 1 = %d\n", game.score[0], game.score[1]);
    return 0;
}

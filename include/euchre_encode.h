#ifndef EUCHRE_ENCODE_H
#define EUCHRE_ENCODE_H

#include "euchre_policy.h"

#include <stdbool.h>
#include <stdint.h>

#define EUCHRE_OBSERVATION_LAYOUT_VERSION 1
#define EUCHRE_OBSERVATION_SIZE 138
#define EUCHRE_ENCODED_ACTION_MASK_SIZE EUCHRE_ACTION_COUNT
#define EUCHRE_ENCODED_NONE (-1)

/*
 * Frozen version-1 offsets. Arrays begin at the named offset and use the
 * dimensions documented beside them.
 */
typedef enum {
    EUCHRE_OBS_VERSION = 0,
    EUCHRE_OBS_PLAYER = 1,
    EUCHRE_OBS_DEALER = 2,
    EUCHRE_OBS_CURRENT_PLAYER = 3,
    EUCHRE_OBS_LEADER = 4,
    EUCHRE_OBS_CALLER = 5,
    EUCHRE_OBS_SITTING_OUT = 6,
    EUCHRE_OBS_GOING_ALONE = 7,
    EUCHRE_OBS_PHASE = 8,
    EUCHRE_OBS_TRUMP = 9,
    EUCHRE_OBS_TURNED_SUIT = 10,
    EUCHRE_OBS_UPCARD = 11,
    EUCHRE_OBS_BID_TURNS = 12,
    EUCHRE_OBS_BID_HISTORY_COUNT = 13,
    EUCHRE_OBS_TRICKS_PLAYED = 14,
    EUCHRE_OBS_TRICK_PLAYS = 15,
    EUCHRE_OBS_TEAM_0_TRICKS = 16,
    EUCHRE_OBS_TEAM_1_TRICKS = 17,
    EUCHRE_OBS_TEAM_0_SCORE = 18,
    EUCHRE_OBS_TEAM_1_SCORE = 19,
    EUCHRE_OBS_HAND_MASK = 20,          /* 24 values */
    EUCHRE_OBS_PLAYED_MASK = 44,        /* 24 values */
    EUCHRE_OBS_BID_HISTORY = 68,        /* 8 records x 5 values */
    EUCHRE_OBS_TRICK_HISTORY = 108,     /* 5 tricks x 4 seats */
    EUCHRE_OBS_TRICK_LEADERS = 128,     /* 5 values */
    EUCHRE_OBS_TRICK_WINNERS = 133      /* 5 values */
} EuchreObservationOffset;

typedef enum {
    EUCHRE_ENCODED_BID_PLAYER = 0,
    EUCHRE_ENCODED_BID_ROUND = 1,
    EUCHRE_ENCODED_BID_ACTION = 2,
    EUCHRE_ENCODED_BID_SUIT = 3,
    EUCHRE_ENCODED_BID_ALONE = 4,
    EUCHRE_ENCODED_BID_WIDTH = 5
} EuchreEncodedBidOffset;

/* Encode a public observation without exposing compiler-dependent struct bytes. */
bool euchre_encode_observation(
    const EuchreObservation *observation,
    int16_t output[EUCHRE_OBSERVATION_SIZE]);

/* Convert C booleans to the frozen byte mask expected by Python. */
bool euchre_encode_action_mask(
    const bool legal_actions[EUCHRE_ACTION_COUNT],
    uint8_t output[EUCHRE_ENCODED_ACTION_MASK_SIZE]);

#endif

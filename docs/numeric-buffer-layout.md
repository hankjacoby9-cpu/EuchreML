# Numeric Buffer Layout

This document freezes the numeric interface between the C environment and
foreign consumers such as Python. Consumers must use the layout version stored
at observation index `0` and reject versions they do not support.

## Representation rules

- Observation type: signed 16-bit integer (`int16_t`)
- Observation length: `138`
- Action-mask type: unsigned 8-bit integer (`uint8_t`)
- Action-mask length: `35`
- Missing or unused categorical value: `-1`
- Boolean and mask values: `0` or `1`
- Card IDs: `suit * 6 + rank`, producing IDs `0` through `23`
- Seats: `0` through `3`; partnerships are seats with the same value modulo 2
- Suits: clubs `0`, diamonds `1`, hearts `2`, spades `3`, no suit `4`
- Ranks: nine `0`, ten `1`, jack `2`, queen `3`, king `4`, ace `5`
- Phases: first bidding round `0`, second bidding round `1`, playing `2`,
  hand complete `3`

## Observation layout version 1

| Range | Length | Meaning |
|---|---:|---|
| `0` | 1 | Layout version, currently `1` |
| `1` | 1 | Observing player |
| `2` | 1 | Dealer |
| `3` | 1 | Current player |
| `4` | 1 | Current trick leader, or `-1` before play |
| `5` | 1 | Caller, or `-1` before trump is selected |
| `6` | 1 | Sitting-out partner, or `-1` |
| `7` | 1 | Going-alone flag |
| `8` | 1 | Game phase |
| `9` | 1 | Trump suit, or no suit before selection |
| `10` | 1 | Turned suit |
| `11` | 1 | Upcard ID |
| `12` | 1 | Passes/actions completed in the current bidding round |
| `13` | 1 | Number of valid bidding-history records |
| `14` | 1 | Completed tricks |
| `15` | 1 | Cards played in the current trick |
| `16` | 1 | Tricks won by team 0 |
| `17` | 1 | Tricks won by team 1 |
| `18` | 1 | Match score for team 0 |
| `19` | 1 | Match score for team 1 |
| `20..43` | 24 | Observing player's hand mask by card ID |
| `44..67` | 24 | Public played-card mask by card ID |
| `68..107` | 40 | Eight bidding records, five values per record |
| `108..127` | 20 | Card ID by trick then seat: `5 x 4` |
| `128..132` | 5 | Leader of each trick |
| `133..137` | 5 | Winner of each trick; unfinished tricks use `-1` |

Each bidding record contains:

| Record offset | Meaning |
|---:|---|
| `0` | Acting player |
| `1` | Bidding round: `1` or `2` |
| `2` | Action: pass `0`, order up `1`, call trump `2` |
| `3` | Suit, or no suit for a pass |
| `4` | Going-alone flag |

Unused bidding records and trick-history entries contain `-1`.

The buffer intentionally excludes opponent hands, hidden kitty cards, the
dealer's discard, and random-generator state.

## Action IDs and mask

The action mask contains one byte for each fixed action ID. A value of `1`
means the action is legal in the current state.

| IDs | Meaning |
|---|---|
| `0..23` | Play or discard the card with that permanent card ID |
| `24` | Pass |
| `25` | Order up |
| `26` | Order up alone |
| `27..30` | Call clubs, diamonds, hearts, or spades |
| `31..34` | Call clubs, diamonds, hearts, or spades alone |

## Compatibility policy

Version 1 offsets and meanings are frozen. Code may fix internal engine logic
without changing the version when the encoded meaning remains the same.

Any change to an existing offset, value meaning, numeric enum mapping, buffer
length, or scalar type requires a new layout version. New versions should use a
new encoder while retaining the old encoder for saved models when practical.

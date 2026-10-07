# Trade engine audit and redesign

Audit of the engine as of commit `f153705` (the "M-HATE" penalty model), and the design that replaced it.

## 1. What the old engine did (data flow)

```
ESPN player ──► base PPG (blend of season avg and projection, actuals capped at 60%)
            ──► weekly[4] = ESPN weekly projection (or base) × matchup ±12% × injury availability; 0 on bye
free agents ──► DRV[pos] = avg of top-3 healthy free agents (× 0.9 QB / × 0.5 RB-WR-TE after #20)

lineup solver: per slot, best rostered player UNLESS the waiver value is higher → waiver plays
starter = in lineup ≥ half the weeks; full-time = ≥ weeks-1

side impact (each side):
  incoming EXCLUDED (count 0, removed before solving) if: 2nd QB/TE behind an S-tier starter,
    or PPG − DRV < 1.0, or starts fewer than weeks-1 weeks
  delta = lineup after − before;  perceived = delta − 50% of any QB gain
  blocked = loses a starter AND every incoming player is redundant

target MAI = perceived − TAP − BCP − TSP − PLT − AP − LAT − TFL
  TAP 20% of each S-tier player's PPG (×2 in packages)   BCP ½ VOR of each bench cut
  TSP 2-for-2 2.0 / target gets more 3.0 / fewer 3.5, on max(nominal, "real pieces")
  PLT ½ (PPG − DRV) for a lost QB1 or S-tier starter     AP 3.0   LAT 1.5   TFL ½ talent gap over 85%
you: effective = delta − sell_low (½ PPG gap when best-in < 85% of best-out)
tiers: NONE if blocked, target delta < 0, or your effective ≤ 0.  Win-Win MAI ≥ 2 & you ≥ 1.5 & 1-1/2-2
       & both solve a hole & no redundancy.  Worth a Shot MAI ≥ .5.  Long Shot −1.5 ≤ MAI < .5 & you ≥ 3.
generation: top-8 by PPG each side, all 1-1 / 2-1 / 1-2 / 2-2, sort by NMU, 3 per tier.
```

## 2. Problems found, ranked

1. **One objection charged four times.** Target gives a star for depth: lineup delta already
   charges the loss, then TAP (×2), TSP, PLT and TFL each charge "you're giving up your best
   player / it's a package" again. ~20 PPG of friction on top of a real −7. The 1-for-1
   Maye/Likely trade still drew TSP 3.5 + PLT 2.8 via the "real pieces" rule.
2. **Waivers set player value.** VOR-vs-waivers decided what counts as a real piece, the size of
   PLT, the lineup floor, redundancy, and the solver benched rostered starters below the waiver
   number (Herbert case). The 10%/50% discounts were patches on this.
3. **QB logic fought itself.** PLT_QB made losing a QB1 *extra* bad; QB_GAIN_HAIRCUT made gaining
   one *less* good. Both existed because the QB waiver level was high.
4. **Hard veto on target lineup delta < 0.** Any lineup dip was "denied" whatever the value
   received. That produced "no reason to accept" on trades the other manager wanted.
5. **Binary exclusions.** Part-time / below-margin / redundant players were set to exactly 0 and
   removed before solving, so they couldn't even cover a bye.
6. **Matchup double count.** ESPN's weekly `projected_points` already price the opponent; the
   engine multiplied by ±12% again.
7. **Generation by "best names".** Top-8 by PPG per side, all combos: most candidates were players
   the other side would bench. Also offered players the target could pick up free.
8. **Sell-low / TFL on raw PPG.** A 25-PPG QB and a 14-PPG WR were compared as if positions were
   equal; no notion of positional value.
9. **TE special-casing** (single-slot redundancy + PLT) layered on the lineup math instead of
   letting scarcity emerge from the league's TE depth.
10. **Starter/bench was binary** (≥ half the weeks), so a 2-of-4 starter and a 4-of-4 starter
    were the same thing, and a 1-of-4 bye-filler was nothing.

## 3. New architecture

Three currencies, each charged once per side:

| Currency | What it represents | How |
|---|---|---|
| **LINEUP** | this month's starting points | solver over the 4-week window. Rostered players always start when available; a waiver pickup fills a slot only if nobody can play it (bye / injury / no player) or the best option is 3+ PPG below the waiver level. |
| **VALUE** | the asset a manager holds | `asset = v (1 + v/10) × availability`, `v = PPG − 0.75 × last-starter level`. Last-starter level = K-th best rostered PPG at the position, K = teams × (slots + flex share), floored at the waiver level. Convex so a star beats two mids. Incoming assets are discounted by how often the side would start them (bench weight 0.25 QB / 0.3 TE / 0.5 others in single-slot formats). The target overvalues what it gives up ×1.15 (endowment). |
| **FRICTION** (target only) | hassle | 3-player deal 1.5, 4-player 2.5; +0.5 per roster spot freed; a sideways same-position swap must be worth 2.5 PPG or the shortfall is charged. |

| **BALANCE** (both sides) | don't fix one spot by hollowing out a weak one | each position's strength = its dedicated starting slots (flex left out) over the window; if a trade leaves a position further below the league average than it was, 0.5 × the extra shortfall is charged. |

```
your gain  = LINEUP(you) + 0.5 × VALUE(you) − weak spot(you)
acceptance = LINEUP(them) + 0.5 × VALUE(them) − friction + freed − weak spot(them)
Win-Win: acceptance ≥ 1.5, you ≥ 1.5, 1-for-1    Worth a Shot: ≥ 0.5, you ≥ min gain
Long Shot: −3.0 ≤ acceptance < 0.5, you ≥ 2.0
every tier: the target needs a concrete reason — lineup +1.0, a weak/empty slot fixed, or value gained    blocked: gives up a starter for nothing it would start
```

What this subsumes: TAP, TFL and sell-low (convex asset value + endowment); PLT and the lineup
floor (solver with next rostered player); BCP (cut players are outgoing assets); TSP, AP and the
package multiplier (one hassle figure + the asset math, which makes consolidation attractive and
star-for-depth unattractive on its own); LAT (sideways rule); QB haircut and PLT_QB (gone: QB
scarcity comes from the QB last-starter level, which is high in 1QB and low in superflex);
redundancy exclusions (bench-weighted incoming value + continuous start share).

Horizons: LINEUP is 4 weeks (matchups, byes, injury status); VALUE uses the season blend and an
8-week injury availability. A single matchup moves the lineup number only.

Generation: each side's candidate pool = the other roster's players who would start for it (ranked
by lineup gain) plus its two best assets (consolidation targets), excluding anyone at or below the
waiver level.

## 4. Old vs new on the known shapes

| case | old (you / MAI / tier) | new (you / acceptance / tier) |
|---|---|---|
| Maye for Likely (they want it) | +2.0 / −10.2 / none | +7.1 / −6.1 / none |
| Star WR for 2 mid RBs (rejected IRL) | +7.0 / −27.1 / none | +13.7 / −19.8 / none |
| 2 RBs for star, they have RB holes | +5.0 / −17.0 / none | +5.7 / −1.2 / Long Shot |
| 1-for-1 hole fill both ways | +2.0 / +3.0 / Win-Win | +2.1 / +4.4 / Win-Win |
| Allen (spare QB) for Evans | −5.6 / +5.7 / none | −11.6 / +17.3 / none |
| RB 14 for their WR2 14, RB would sit | 0.0 / −11.6 / none | −0.8 / −5.5 / none |
| QB 16.5 for QB 15 sideways | −1.5 / −0.8 / none | −2.9 / +1.5 / none |

Scores are now in a range a person can read (−20..+20 rather than −27), the one consolidation
that fills real holes becomes a Long Shot instead of −17, and the Maye/Likely verdict is "they'd
give up the best player in the deal and lose 1.3 PPG" rather than a structure penalty on a 1-for-1.

## 5. What the engine still cannot model

- **Breakouts.** Base PPG caps actuals at 60%, so a player outscoring his projection is under-rated
  for weeks. The Maye trade is probably this: the other manager sees a top-8 QB by season, the
  engine sees a 15.9 blend. Rest-of-season projections from ESPN would fix it.
- **Manager-specific taste.** Favourite players, "I never trade with him", dynasty-style youth
  preference. Not observable from the roster.
- **Playoff schedule and bye stacking** beyond the 4-week window.
- **League-wide market.** The target's alternatives (other offers) aren't modelled; acceptance is
  absolute, not relative.
- **Calibration data is thin.** Seven rejected packages and a handful of entertained 1-for-1s.
  HASSLE, ENDOWMENT, CONVEXITY and VALUE_WEIGHT are reasoned, not fitted.

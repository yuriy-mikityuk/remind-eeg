# N-of-1 series: cue, sham and no-cue nights

A cued-vs-uncued delta from one night mixes two things: what the cues did, and what a
night with sounds does. A series of nights in three arms separates them a little:

- `cue`: the night plays its cues as planned, like any Pilot 4 or 5 night.
- `sham`: an untrained sound plays wherever a cue would, at the same times, lengths and
  volumes. Only the sound differs.
- `no_cue`: nothing plays. The scheduler logs a `skip` with `metadata.virtual_play`
  wherever the cue arm would have played.
- `washout`: a night between blocks without a cueing run.

The one randomised, participant-blind home TLR study interleaved untrained-sound and
no-cue nights with TLR nights in this way (Konkoly et al. 2024, in
[`references.md`](references.md)). This page is the runbook for such a series with
REMind. It describes what was run, not whether cueing works: at n = 1 the experimenter
cannot be blind, a sham still makes sound, the untrained sound may be recognised, and
every output is a single-case description.

## 1. Plan and seal the series

```bash
muse-tmr plan-night-schedule --blocks 2 --washout-nights 1
```

Each block has one night of each arm. Within every group of three blocks each arm takes
each place in its block once, so no arm is always the first night. Washout nights sit
between blocks. The order comes from `--seed`, or from a seed drawn at random and kept
only in the file; the command prints neither the seed nor an arm.

It writes two files under the gitignored `data/protocol/night_schedules/`:

- `<schedule-id>.json`, the schedule: arms, seed and a random salt. Do not open it
  before the series ends.
- `<schedule-id>.sealed.json`, its sealed companion. It holds only `commitment_sha256`,
  the sha256 of the schedule's canonical JSON followed by the salt, so you can share or
  preregister it before the first night without giving an arm away.

The printed `nights=` line lists the night IDs in order (`night-001`, ...). Name each
night's puzzle session after its night ID (`data/protocol/night-001_puzzles.json` with
session ID `night-001`), so the morning files and the night's summary carry the same
name.

`--sham-cue-id` (default `sham_soft_tone`) names the untrained sound. It must be in the
night's cue library with `protocol: "sham"`, and it must not be a trained cue: not the
cue of any puzzle in the catalog, not a TLR cue, and not a generated tone of the same
frequency as one. `muse-tmr create-cue-library` writes a starter library that has
`sham_soft_tone`, a 330 Hz tone that no other starter cue uses. A runner checks the sham
cue on every night of the series, whatever its arm, so a refusal says nothing about the
arm.

`muse-tmr show-night-schedule <schedule.json>` shows the nights, their blocks and the
washout nights, and checks the schedule against its companion. While the series is
sealed it prints `sealed` for every other night.

Before the first night, write the preregistration:

```bash
muse-tmr prereg-template data/protocol/night_schedules/<schedule-id>.json \
  --output data/protocol/night_schedules/<schedule-id>_prereg.md
```

It is OSF-style Markdown made from the schedule's design (blocks, arms, washout nights,
the night IDs in order and how the order is drawn), the sealed `commitment_sha256` and
the protocol block: the one of `--protocol-preset` (default `remind_default_v1`, as for
every runner) or of a file given with `--protocol-metadata`, such as a night's puzzle
session. It never prints or writes an arm, the seed or the salt, so it can be filed, on
OSF for example, before the first night: two schedules of one design drawn on the same
day differ in it only in their commitment. Fill in what it marks _To fill in_: the
hypotheses, whether the seed was yours or drawn at random, the primary outcome and
contrast, the stopping rule and the exclusions.

It also says how many arm orders the blocks allow, which decides what the analysis can
show (section 7). Two or three blocks allow 12, too few for any difference to fall
outside its interval; four allow 72. Plan the blocks with that in mind. A template
written after unsealing says it is not a preregistration.

## 2. Each night

Set up the night as Pilot 5 does (`docs/pilot5_full_night.md`): puzzles, assignment,
TLR training, calibration. Then start it with the schedule and the night's ID:

```bash
muse-tmr run-pilot5-full-night \
  --night-schedule data/protocol/night_schedules/<schedule-id>.json \
  --night-id night-001 \
  ...the usual Pilot 5 flags...
```

`run-pilot4-cueing` and `simulate-replay-cues` take the same two flags. The runner checks
the schedule against its sealed companion before it reads anything else, refuses a
washout night and a night the schedule does not hold, and exits with code 2 if the
schedule changed after it was sealed. It never prints the arm: on a night with an arm its
last line leaves out the play counts, the first play, the volumes and the scorer's
status, which would give the arm away, and says where they are.

`--arm cue|sham|no_cue` runs a night as one arm without a schedule, open label, for a
replay or a smoke test. `--night-id` then only labels the night, and `--sham-cue-id`
replaces the sham cue of either.

The arm changes what a cue becomes, never when one comes or how loud. The gate, the
arousal guard, the scheduler's timing and limits and the protocol preset are the same in
every arm, and so is the night's `protocol` block. A sham play keeps the length and the
volume hint of the cue it stands in for and plays through the same `AudioCuePlayer`,
backend and calibrated cap; a no-cue night sends nothing to the player.

Leave everything the night wrote closed until the series is unsealed (section 6), not
just until the morning: `pilot5_summary.json`, `metadata.json`, the logs, and the
`cue_arousal_report.*` and `protocol_fidelity.*` files. They name the arm (`arm`,
`night_arm`, reasons such as `sham_arm`) and the cued puzzles (`puzzle_id` on every play).
Every block holds one night of each arm, so the arms of the nights already run narrow
down the ones still to come, and a night's cued puzzles would unblind its delayed
retests. The runner's last line names no arm and says whether the night passed. If
something makes you open a night's files early, note when: the nights after that are not
blind.

## 3. The morning

Use the app's Morning tab or the CLI as in Pilot 5. The page never learns which puzzles
were cued or which arm the night had: it reads no schedule and no night log, and the sound
check plays every puzzle's sound in an order made from the session ID alone, the same in
every arm.

The dream report (schema 2) asks, besides lucidity and recall:

- whether a sound was heard, and where: `not_perceived`, `in_dream`, `awake`, `unsure`
  (heard, but not sure whether dreaming) or `unknown`;
- whether a sound woke the sleeper: `yes`, `no` or `unknown`;
- the times of any eye-signal attempts (left-right-left-right), as marked by the
  sleeper. Nothing detects or checks an eye signal.

```bash
muse-tmr record-dream-report data/protocol/night-001_puzzles.json \
  --output data/reports/night-001_dream_report.json \
  --lucid no --cue-perception in_dream --woke-from-sound no \
  --eye-signal-attempt 04:20 \
  --confidence 0.6 --dream-text "..."
```

## 4. Delayed retests

Effects of cueing during sleep have shown up only at a delayed test in some studies: a
week later for cueing in REM sleep (Pereira et al. 2023), ten days later for cueing in
NREM sleep (Rakowska et al. 2021). So retest the same puzzles again then. A delayed retest is blind like the morning one and saves its own
file:

```bash
muse-tmr record-puzzle-retest data/protocol/night-001_puzzles.json \
  --catalog data/protocol/puzzle_catalog.json \
  --assignment data/protocol/night-001_assignment.json \
  --wave day7 \
  --output data/reports/night-001_retest_day7.json \
  ...the results...
```

In the Morning tab pick "A week later" or "Ten days later" above the retest; it saves
`<night>_retest_day7.json` or `<night>_retest_day10.json` next to the morning's
`<night>_retest.json`.

## 5. Blind coding of the dream reports

The puzzles the sleeper links in the dream report are self-report. For a blind code,
export each night's packet once its dream report is saved:

```bash
muse-tmr export-blind-coding data/protocol/night-001_puzzles.json \
  --dream-report data/reports/night-001_dream_report.json \
  --catalog data/protocol/puzzle_catalog.json \
  --output data/reports/night-001_blind_coding_packet.json \
  --markdown-output data/reports/night-001_blind_coding_packet.md
```

The packet holds the dream text and the night's puzzles, ID and prompt, in the order the
retest numbers them, which the cued/uncued split does not touch. It holds no cue ID, no
cue condition, no assignment, no arm, nothing the sleeper said about sounds and none of
their links. A rater who codes `sound_cue` needs the puzzles' sounds, which the packet
leaves out with everything else about cues: hand them over with the puzzle materials.

The rater codes every puzzle in any, all or none of three categories, as the blind
raters of Konkoly et al. 2026 did: `sound_cue` (the report has a sound that fits the
puzzle's sound), `direct_conceptual` (the puzzle itself: its objects, words or task) and
`indirect_conceptual` (something clearly related to it, without the puzzle itself):

```bash
muse-tmr record-incorporation-coding data/reports/night-001_blind_coding_packet.json \
  --rater-id rater-a \
  --code p1=direct_conceptual --code p2=none --code p3=sound_cue,indirect_conceptual --code p4=none \
  --output data/reports/night-001_coding_rater-a.json
```

Every puzzle of the packet needs its `--code`, `none` included: a puzzle left out is not
a no. With a second rater, `compare-incorporation-codings` gives percent agreement and
Cohen's kappa per category, and for any category, over the puzzles both coded, on one
night or many:

```bash
muse-tmr compare-incorporation-codings data/reports/night-001_coding_rater-a.json \
  data/reports/night-001_coding_rater-b.json --output data/reports/night-001_coding_agreement.json
```

The analyses prefer rater codes to the sleeper's links. With more than one rater a
puzzle counts as incorporated when more than half of them put it in some category, so
with two raters both must; the puzzles they split on are listed. At n = 1 the sleeper may
code their own packets before the series is unsealed: blind to which puzzles were cued,
not to the dream. A second person's codes are better.

## 6. Unsealing

After the last delayed retest of the last night:

```bash
muse-tmr unseal-night-schedule data/protocol/night_schedules/<schedule-id>.json
```

It checks the schedule against its sealed companion, records `unsealed_at_utc` in the
schedule (the first time only) and prints the arms. A schedule that changed after it was
sealed is not unsealed, and the command exits with code 1. A night run after the
unsealing records it in its `metadata.json` (`night_arm.schedule_unsealed_at_utc`).

`protocol-fidelity` and `score-cue-arousal` run on every arm. On a sham night they read
the sham plays like any others and add `sham_arm`; on a no-cue night they find no cues,
say `no_cue_arm`, and the fidelity summary counts the night's `virtual_plays`. Both
print the arm, so run them on the nights of a series only after unsealing. That holds for
scoring a night again after `log-pilot4-awakening` too: log the awakenings in the
morning, which prints no arm, and score them after unsealing.

`analyze-cued-uncued` reads one night. On a sham night it counts the sham plays apart,
never as plays of the cued puzzles' sounds, though each keeps its puzzle; with
`--scheduler-events` it prints the arm, so in a series run it after unsealing too.

## 7. The analysis

After unsealing:

```bash
muse-tmr analyze-nof1 data/protocol/night_schedules/<schedule-id>.json \
  --primary-outcome cued_minus_uncued_solve_rate_day7 \
  --output data/reports/<schedule-id>_nof1.json \
  --markdown-output data/reports/<schedule-id>_nof1.md
```

It refuses a schedule that is still sealed, and names `unseal-night-schedule`, and one
that changed after it was sealed. It finds each block night by its ID: the puzzle
session and the assignment in `data/protocol` (`--protocol-dir`); the dream report, the
retests, the sound check and the coding sheets in `data/reports` (`--reports-dir`); and
the `pilot4_summary.json` or `pilot5_summary.json` under `data/recordings`
(`--recordings-dir`) whose `night_id` and `schedule_commitment_sha256` are the series'.
`--summary` names a night's summary, or a Pilot 3 report, and `--coding` a coding sheet
kept elsewhere. Two files of one kind for one night stop it, so a night is never picked
by chance; a second series needs other night IDs (`plan-night-schedule --night-prefix`).

For every night, and as a mean per arm, it gives what the files hold:

- `lucid`, `cue_perceived` and `woke_from_sound` from the dream report;
- `awakening_markers` from the night's `awakening_events.jsonl`;
- `incorporation_rate`, by the rater codes when the night has them, else by the
  sleeper's links;
- `cued_minus_uncued_solve_rate_morning`, `_day7` and `_day10`, as `analyze-cued-uncued`
  computes them for each wave. On a sham or no-cue night the cued puzzles' sounds did
  not play, which is the comparison the arms are for;
- plays and virtual plays from the summary, and the arousal proxies per cue of
  `score-cue-arousal` (`arousal_proxy_per_cue`, from the summary's `cue_arousal` block or
  the `cue_arousal_report.json` next to it, when that report scored this summary or
  Pilot 3 report). Only nights with sound have them, so they stay a mean per arm.

For each outcome and each pair of arms it gives the difference of the means with its
permutation interval. The arm labels are moved within their blocks the way
`plan-night-schedule` draws them: over every order it could have drawn, when there are
at most `--permutations` (10,000) of them, else over that many orders drawn with
`--seed` (0). The interval is the range of those differences once the lowest and the
highest 2.5 % are cut: what the difference looks like when the labels carry no
information. With two or three blocks there are 12 orders and nothing is cut, so no
difference can fall outside; with four there are 72. An outcome with few levels, such
as a yes or no per night or the share of four puzzles, gives many orders the same
difference, so even a large one can sit on the edge of its interval rather than
outside it. No p-value is computed. A difference outside its interval describes this
series, it is not evidence that cueing works, and with eight outcomes and three pairs
of arms some may fall outside by chance: `--primary-outcome` and `--primary-contrast`
mark the one you preregistered.

It also gives each outcome's lag-1 autocorrelation once the arm means are taken out, and
flags the series' first night, with every mean also given without it. The limitations
say what limits the reading: `single_subject`; `no_sham_nights` and `no_no_cue_nights`;
`unbalanced_arms`; `small_n`, fewer than `--min-nights-per-arm` (5) nights with data in
an arm; `unblinded_early`, the schedule unsealed before the series' last morning form or
delayed retest was saved, or before a night started; `sham_detectable`;
`carry_over_possible`; `first_night_effect_possible`; `missing_delayed_retest`;
`missing_arousal_scoring`; `missing_night_data`, a night without its summary, puzzle
session, assignment, dream report or morning retest; `arm_mismatch`, a summary that names
another arm than the schedule, whose arm is kept; `inherits_detector_limits`, the
detector's open issues #116 and #167; and `synthetic_input`.

The Markdown report ends with a checklist of SCRIBE 2016 and CENT 2015
([`references.md`](references.md)): each item is `auto` when the report holds it,
`partial` when it holds part of it, and `manual` when it is yours to write. The analysis
is pure Python, gives the same output for the same seed, and copies no dream text.

Keep everything a series writes under `data/` and out of git: the schedule, its
preregistration, the nights, the reports, the retests, the packets, the coding sheets
and the analysis.

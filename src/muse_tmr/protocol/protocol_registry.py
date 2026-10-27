"""The protocol presets REMind knows, by name (#231).

Every preset names the paper and arm its parameters come from ("parameters from
<paper, arm>"); none of them is the protocol of a lab, and none claims that a night run
with it does what the paper found. ``remind_default_v1`` is today's dataclass defaults
and stays the default of every runner. ``docs/protocol_presets.md`` has the same
content as prose.

Changing anything in a preset changes its ``content_hash()``: raise its ``version`` and
renew the pinned hash in ``tests/test_protocol_presets.py`` in the same commit.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from muse_tmr.protocol.protocol_presets import (
    Citation,
    CueSpec,
    CueTiming,
    IntensityRamp,
    PauseRules,
    PlatformDeviation,
    ProtocolPreset,
    RunnerSettings,
)

PROTOCOL_REGISTRY_VERSION = 1
DEFAULT_PROTOCOL_PRESET = "remind_default_v1"


class ProtocolPresetNotRunnable(ValueError):
    """The preset documents a protocol REMind's runners cannot run."""


# --- citations ------------------------------------------------------------------

_KONKOLY_2021 = dict(
    authors="Konkoly KR, Appel K, Chabani E, et al.",
    year=2021,
    title="Real-time dialogue between experimenters and dreamers during REM sleep",
    venue="Current Biology 31(7):1417-1427.e6",
    doi="10.1016/j.cub.2021.01.026",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8162929/",
    source_check="verified_full_text",
)
_KONKOLY_2024 = dict(
    authors="Konkoly KR, Whitmore NW, Mallett R, Mazurek CY, Paller KA",
    year=2024,
    title="Provoking lucid dreams at home with sensory cues paired with pre-sleep cognitive training",
    venue="Consciousness and Cognition 125:103759",
    doi="10.1016/j.concog.2024.103759",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11542932/",
    source_check="verified_full_text",
)
KONKOLY_2026 = Citation(
    authors="Konkoly KR, Morris DJ, Hurka K, Martinez AM, Sanders KEG, Paller KA",
    year=2026,
    title="Creative problem-solving after experimentally provoking dreams of unsolved puzzles during REM sleep",
    venue="Neuroscience of Consciousness 2026(1):niaf067",
    doi="10.1093/nc/niaf067",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12875123/",
    source_check="verified_full_text",
    note="Read in the PMC copy; the publisher's page is https://academic.oup.com/nc/article/2026/1/niaf067/8456489.",
)
BORGHESE_2022 = Citation(
    authors="Borghese F, Henckaerts P, Guy F, Perez Mayo C, Delplanque S, Schwartz S, Perogamvros L",
    year=2022,
    title="Targeted Memory Reactivation During REM Sleep in Patients With Social Anxiety Disorder",
    venue="Frontiers in Psychiatry 13:904704",
    doi="10.3389/fpsyt.2022.904704",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9281560/",
    source_check="verified_full_text",
)
ESFAHANI_2024 = Citation(
    authors="Jafarzadeh Esfahani M, Salvesen LAC, Picard-Deland C, et al.",
    year=2024,
    title=(
        "Highly effective verified lucid dream induction using combined cognitive-sensory training "
        "and wearable EEG: a multi-centre study"
    ),
    venue="bioRxiv 2024.06.21.600133, version 1, preprint, not peer reviewed",
    doi="10.1101/2024.06.21.600133",
    url="https://www.biorxiv.org/content/10.1101/2024.06.21.600133v1.full",
    source_check="verified_full_text",
    note=(
        "Version 1 was read. Version 2 (2026) lists Picard-Deland first, is titled 'Inducing lucid "
        "dreaming with multisensory stimulation: a preregistered multi-center study' and was not read."
    ),
)
SCHWARTZ_2022 = Citation(
    authors="Schwartz S, Clerget A, Perogamvros L",
    year=2022,
    title="Enhancing imagery rehearsal therapy for nightmares with targeted memory reactivation",
    venue="Current Biology 32:4808-4816.e4",
    doi="10.1016/j.cub.2022.09.032",
    url="https://doi.org/10.1016/j.cub.2022.09.032",
    source_check="notes_only",
    note="From the abstract, as checked in verify_tmr_tlr_science.md row 15; the full text was not read.",
)
SANDERS_2019 = Citation(
    authors="Sanders KEG, Osburn S, Paller KA, Beeman M",
    year=2019,
    title="Targeted Memory Reactivation During Sleep Improves Next-Day Problem Solving",
    venue="Psychological Science 30(11):1616-1624",
    doi="10.1177/0956797619873344",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6843748/",
    source_check="verified_full_text",
    note=(
        "A correction was published (Psychological Science 31(8):1048, 2020); its page answered with a "
        "browser check and its text was not read."
    ),
)

# --- deviations shared by several presets ------------------------------------------

_LEVEL_IS_GAIN = PlatformDeviation(
    parameter="cue level",
    published="a sound level or a level set by hand during the night",
    remind="a software gain from 0 to 1, capped by the calibrated comfortable volume and the 0.20 hard cap; no preset sets it",
    reason="the player has no sound-level calibration, and sleep-time volume stays behind the calibration",
)
_NO_RAMP = PlatformDeviation(
    parameter="intensity ramp",
    published="cue intensity raised step by step while sleep stayed undisturbed",
    remind="every cue of a night at the same gain, lowered only by the arousal guard's lower_volume",
    reason="automatic ramping is out of scope (#231); the fidelity report measures it instead",
)
_GATE_REM = PlatformDeviation(
    parameter="REM",
    published="REM scored by experts or a validated headband in real time",
    remind="gate-open runs of REMind's detector on forehead EEG, no PSG, no EOG",
    reason="Muse S Athena; the detector is checked against an automated stager only",
)
_FIRST_CUE_LATENCY = PlatformDeviation(
    parameter="first cue",
    published="cues began within the first REM epoch or seconds after REM onset",
    remind="the gate's 60 s stable window: the first cue comes 60 to 85 s after a REM onset at a 30 s stride (synthetic test, #167)",
    reason="30 s epochs and the stable window; presets never lower the window below 60 s",
)
_TLR_TIMING = PlatformDeviation(
    parameter="TLR cue spacing",
    published="TLR cues a fixed number of seconds apart",
    remind=(
        "the scheduler emits the TLR cues an epoch decision makes due, and the night plays them then, "
        "so cues planned closer than the stride play back to back in one epoch"
    ),
    reason="the scheduler runs once per epoch; --stride-seconds at or below the TLR interval comes closer",
)
_ONE_CUE_PER_PUZZLE = PlatformDeviation(
    parameter="puzzle cue repeats",
    published="each cued sound repeated many times per REM period",
    remind="the scheduler plays each cued puzzle's cue once per night, in order, at most max_puzzle_cues_per_block per REM block",
    reason="TmrCueScheduler, unchanged here",
)
_GENERATED_TONES = PlatformDeviation(
    parameter="puzzle sounds",
    published="a recorded sound per puzzle",
    remind="Pilot 4 and 5 play generated_tone cues only, one sine tone per puzzle",
    reason="AudioCuePlayer plays generated tones",
)


def _deviations(*items: PlatformDeviation) -> Tuple[PlatformDeviation, ...]:
    return tuple(items)


# --- presets ------------------------------------------------------------------------

REMIND_DEFAULT_V1 = ProtocolPreset(
    name="remind_default_v1",
    version=1,
    title="REMind's own defaults (repository default, no published basis)",
    runnable=True,
    design_note=(
        "Today's dataclass defaults, frozen as a preset so a night can name them. None comes from a paper. "
        "The 60 s stable window and the 120 s cooldowns were set by hand; puzzle cues at least 30 s apart and "
        "at most 4 per REM block are the scheduler's starting values. The TLR tone of 396 Hz for 1 s, played "
        "3 times 8 s apart, began as a starter tone for smoke tests (commit b932d08, 2026-05-12) and no reason "
        "for 396 Hz is recorded; the cited presets replace it with the paper's cue where a tone can stand in."
    ),
    caveats=("No published basis for any value.",),
    cues=(
        CueSpec("tlr", "one generated sine tone", duration_seconds=1.0, frequencies_hz=(396.0,)),
        CueSpec("puzzle", "each puzzle's own cue from the cue library"),
    ),
    timing=(
        CueTiming("tlr", interval_seconds=(8.0, 8.0), max_cues_per_rem_period=3, note="one block per REM block"),
        CueTiming("puzzle", interval_seconds=(30.0, None), max_cues_per_rem_period=4),
    ),
    pauses=PauseRules(
        start_after_rem_seconds=60.0,
        resume_after_rem_seconds=60.0,
        pause_seconds=(("arousal", 120.0),),
        stops_on=("rem_gate_closed",),
        note="The arousal guard pauses 120 s; when the gate closes the scheduler waits its 120 s cooldown.",
    ),
    ramp=IntensityRamp(start="the cue's volume hint or the default volume, under the calibration cap"),
    runner=RunnerSettings(),
)

KONKOLY2021_NORTHWESTERN = ProtocolPreset(
    name="konkoly2021_northwestern",
    version=1,
    title="Parameters from Konkoly et al. 2021, Northwestern arm (USA group)",
    runnable=True,
    design_note=(
        "The lucidity cue and its timing in REM from the USA group's procedure: three rising beeps, "
        "presented about every 30 s once REM was scored and while it stayed stable, about 10 times before "
        "the math problems began. REMind plays them as its TLR block."
    ),
    citations=(Citation(arm="Northwestern University (Procedures for Group in the USA)", **_KONKOLY_2021),),
    caveats=(
        "Morning naps of 22 participants who remembered at least one dream a week; the paper is about "
        "two-way communication in lucid REM, not about how often cues induce lucidity.",
        "One participant heard a 1000 ms violin sound with flashing red lights instead of the beeps.",
    ),
    cues=(
        CueSpec(
            "tlr",
            "three pure-tone beeps rising in pitch, about 40 to 45 dB SPL",
            duration_seconds=0.65,
            frequencies_hz=(400.0, 600.0, 800.0),
        ),
    ),
    timing=(
        CueTiming(
            "tlr",
            interval_seconds=(30.0, 30.0),
            max_cues_per_rem_period=10,
            note="about every 30 s while REM stayed stable; about 10 cues without a lucid signal, then math problems",
        ),
    ),
    pauses=PauseRules(
        stops_on=("rem_unstable", "lucid_signal"),
        note="Cues began once REM was scored (no stable time is given) and went on while REM stayed stable.",
    ),
    ramp=IntensityRamp(level_db_spl=(40.0, 45.0)),
    training=(
        "Targeted lucidity reactivation before the nap, derived from Carr et al.: the cue up to 15 times "
        "over 15 min while falling asleep, the first 4 followed by spoken guidance to become lucid."
    ),
    runner=RunnerSettings(
        tlr_repetitions=10,
        tlr_interval_seconds=30.0,
        tlr_tone_frequency_hz=600.0,
        tlr_tone_duration_seconds=0.65,
    ),
    deviations=_deviations(
        PlatformDeviation(
            parameter="TLR cue",
            published="three pure-tone beeps rising 400, 600 and 800 Hz, about 650 ms in all",
            remind="one generated 600 Hz tone of 0.65 s",
            reason="a generated_tone cue has one frequency",
        ),
        _LEVEL_IS_GAIN,
        _GATE_REM,
        _FIRST_CUE_LATENCY,
        PlatformDeviation(
            parameter="after the cues",
            published="after a lucid eye signal or about 10 cues, math problems for two-way communication",
            remind="no eye-signal detection; the TLR block ends after its 10 cues and puzzle cues may follow",
            reason="REMind is a one-way protocol",
        ),
    ),
)

KONKOLY2021_NIJMEGEN = ProtocolPreset(
    name="konkoly2021_nijmegen",
    version=1,
    title="Parameters from Konkoly et al. 2021, Nijmegen arm (Netherlands group)",
    runnable=True,
    design_note=(
        "The Netherlands group's REM cueing: TLR cues after the first 30 s epoch of REM, one every 10 to 15 s, "
        "up to 10 per REM period, paused while the EEG showed an arousal. These are the timings the research "
        "report filed under the Paris arm; the STAR Methods give them for the Netherlands group "
        "(konkoly2021_paris has the Paris procedure)."
    ),
    citations=(Citation(arm="Donders Institute, Nijmegen (Procedures for Group in the Netherlands)", **_KONKOLY_2021),),
    caveats=(
        "Morning naps of 13 participants with high-density EEG; the TLR cues were beeps together with "
        "blinking lights, on white noise of at most 45 dBA.",
        "The tones' frequency and length are not restated for this group; the stand-in follows the USA procedure "
        "the group refers to.",
    ),
    cues=(CueSpec("tlr", "beeping tones with blinking lights, as in the USA group's TLR training"),),
    timing=(
        CueTiming("tlr", interval_seconds=(10.0, 15.0), max_cues_per_rem_period=10, note="one cue every 10 to 15 s"),
    ),
    pauses=PauseRules(
        start_after_rem_seconds=30.0,
        pause_until_clear=("arousal",),
        note="Cues were paused if the EEG showed an arousal and resumed at REM onset.",
    ),
    ramp=IntensityRamp(background_noise="white noise at most 45 dBA"),
    training="15 min of TLR training before the nap as in the USA procedure, the cues 1 min apart.",
    runner=RunnerSettings(
        tlr_repetitions=10,
        tlr_interval_seconds=15.0,
        tlr_tone_frequency_hz=600.0,
        tlr_tone_duration_seconds=0.65,
    ),
    deviations=_deviations(
        _FIRST_CUE_LATENCY,
        _TLR_TIMING,
        PlatformDeviation(
            parameter="arousal pause",
            published="paused while the EEG showed an arousal, resumed at REM onset",
            remind="the arousal guard's motion, alpha, heart-rate and artifact proxies pause cueing for 120 s",
            reason="forehead EEG and IMU proxies, no scored arousals",
        ),
        PlatformDeviation(
            parameter="TLR cue",
            published="beeping tones with blinking lights",
            remind="one generated 600 Hz tone of 0.65 s, no light",
            reason="a generated_tone cue has one frequency, and REMind has no light output",
        ),
        _LEVEL_IS_GAIN,
        _GATE_REM,
    ),
)

KONKOLY2021_OSNABRUECK = ProtocolPreset(
    name="konkoly2021_osnabrueck",
    version=1,
    title="Parameters from Konkoly et al. 2021, Osnabrück arm (Germany group)",
    runnable=True,
    design_note=(
        "Only the start and stop rules transfer: stimulation began after 5 min of stable REM and stopped on an "
        "arousal or NREM. The stimuli themselves were Morse-coded math problems for experienced lucid dreamers "
        "to answer, not lucidity cues; REMind plays its own TLR and puzzle cues under these rules."
    ),
    citations=(Citation(arm="Osnabrück University (Procedures for Group in Germany)", **_KONKOLY_2021),),
    caveats=(
        "10 experienced lucid dreamers, woken after 10 min of REM 4.5 h after sleep onset and kept awake 45 min "
        "before the stimulation periods.",
    ),
    cues=(
        CueSpec(
            "any",
            "Morse-coded math problems as pure tones at 470 or 600 Hz (300 ms dots, 900 ms dashes) or LED flashes",
            frequencies_hz=(470.0, 600.0),
        ),
    ),
    timing=(CueTiming("any", note="a 20 s answer pause after each problem; no cue interval is given"),),
    pauses=PauseRules(
        start_after_rem_seconds=300.0,
        stops_on=("arousal", "nrem", "response"),
        note="Stimulation stopped when the sleeper answered, had an arousal or went into NREM.",
    ),
    ramp=IntensityRamp(start="not given", rises=True, on_arousal="stimulation stopped", cap="raised for each new problem"),
    training="Home training to decode Morse-coded problems; in the lab, 45 min awake practice after a REM awakening.",
    runner=RunnerSettings(gate_min_stable_seconds=300.0),
    deviations=_deviations(
        PlatformDeviation(
            parameter="stimuli",
            published="Morse-coded math problems to answer with eye movements",
            remind="REMind's TLR block and puzzle cues",
            reason="REMind has no two-way channel; only the timing rules are taken",
        ),
        PlatformDeviation(
            parameter="stop on arousal",
            published="stimulation stopped for the period",
            remind="the arousal guard pauses 120 s and cueing can resume in the same REM period",
            reason="the scheduler pauses on a guard pause; the fidelity report counts cues after one",
        ),
        _NO_RAMP,
        _LEVEL_IS_GAIN,
        _GATE_REM,
    ),
)

KONKOLY2021_PARIS = ProtocolPreset(
    name="konkoly2021_paris",
    version=1,
    title="Parameters from Konkoly et al. 2021, Paris arm (France group), documentation only",
    runnable=False,
    not_runnable_reason=(
        "Stimulation started 1 min into a daytime nap while the participant was still awake and ran in fixed "
        "blocks whatever the stage; REMind cues only in gate-open REM periods, so this timing cannot run here."
    ),
    design_note=(
        "Kept so the Paris arm is not confused with the Nijmegen timings: one participant with narcolepsy, "
        "stimuli in blocks of 6 every 10 s with a jitter, 1 min of white noise between blocks, 20 min in all."
    ),
    citations=(Citation(arm="Paris Brain Institute (Procedures for Group in France)", **_KONKOLY_2021),),
    caveats=("One participant with narcolepsy type 1 who answered with facial muscle contractions.",),
    cues=(CueSpec("any", "spoken French words (up, down, mixed) in a female voice; earlier, taps, tones and lights"),),
    timing=(
        CueTiming(
            "any",
            interval_seconds=(10.0, 10.0),
            max_cues_per_rem_period=6,
            note="6 stimuli per block, every 10 s with a jitter; 10 blocks with 1 min OFF between them",
        ),
    ),
    pauses=PauseRules(note="No pause rule; the blocks ran on a fixed schedule."),
    runner=RunnerSettings(),
)

KONKOLY2024_EXP1 = ProtocolPreset(
    name="konkoly2024_exp1",
    version=1,
    title="Parameters from Konkoly et al. 2024, Experiment 1 (home app)",
    runnable=True,
    design_note=(
        "A phone app without sleep staging: cues from 6 h after the evening TLR training, every 20 to 40 s, "
        "each 0.16 % louder than the last, paused for 5 min after a movement and for 45 min after the sleeper "
        "reported that a cue woke them. REMind gates the same cue on REM, which the authors name as the obvious "
        "improvement, so this is the home schedule moved onto the REM gate."
    ),
    citations=(Citation(arm="Experiment 1", **_KONKOLY_2024),),
    caveats=(
        "No sleep staging: cueing ran on the time of night and the phone's motion sensor.",
        "19 completers and no sham condition in this experiment.",
    ),
    cues=(
        CueSpec(
            "tlr",
            "three pure-tone beeps rising in pitch lasting about 650 ms, or a 1000 ms violin sound",
            duration_seconds=0.65,
            frequencies_hz=(400.0, 600.0, 800.0),
        ),
    ),
    timing=(
        CueTiming("tlr", interval_seconds=(20.0, 40.0), note="from 6 h after training ended, for the rest of the night"),
    ),
    pauses=PauseRules(
        pause_seconds=(("movement", 300.0),),
        pause_after_awakening_seconds=2700.0,
        note=(
            "Paused until 5 min after movement ceased, and until 45 min after the sleeper said a cue woke them "
            "and went back to sleep; cueing then restarted at the discrimination threshold."
        ),
    ),
    ramp=IntensityRamp(
        start="the perceptual discrimination threshold",
        rises=True,
        step_percent_per_cue=0.16,
        cap="none: the intensity kept rising until the sleeper woke",
    ),
    training=(
        "Each evening the app paired the cue with a lucid mindset: 4 cues with spoken guidance, then 12 more "
        "over 15 min at 45 to 120 s intervals; shorter on nights 2 to 7."
    ),
    runner=RunnerSettings(
        tlr_interval_seconds=30.0,
        scheduler_cooldown_seconds=300.0,
        tlr_tone_frequency_hz=600.0,
        tlr_tone_duration_seconds=0.65,
    ),
    deviations=_deviations(
        PlatformDeviation(
            parameter="when cues start",
            published="6 h after training, whatever the stage",
            remind="only in gate-open REM periods",
            reason="REMind is REM-gated",
        ),
        PlatformDeviation(
            parameter="cues per REM period",
            published="no limit: cues every 20 to 40 s for the rest of the night",
            remind="a TLR block of 3 cues per REM block (repository default, not from the paper)",
            reason="the TLR block has a fixed length",
        ),
        PlatformDeviation(
            parameter="movement pause",
            published="5 min after movement ceased",
            remind="300 s scheduler cooldown, which every arousal guard pause lasts at least",
            reason="the guard pauses on its own motion proxy",
        ),
        PlatformDeviation(
            parameter="awakening pause",
            published="45 min after a reported cue-caused awakening",
            remind="not applied during the night: awakening markers are logged, not read by the scheduler",
            reason="the fidelity report checks the markers against the next cue",
        ),
        _NO_RAMP,
        _LEVEL_IS_GAIN,
        PlatformDeviation(
            parameter="TLR cue",
            published="three rising beeps or a violin sound",
            remind="one generated 600 Hz tone of 0.65 s",
            reason="a generated_tone cue has one frequency",
        ),
    ),
)

KONKOLY2024_EXP2 = ProtocolPreset(
    name="konkoly2024_exp2",
    version=1,
    title="Parameters from Konkoly et al. 2024, Experiment 2 (public app)",
    runnable=True,
    design_note=(
        "Experiment 1's app with two harp melodies of about 3 s that faded in and out, a 1 min movement pause, "
        "and the intensity capped where the sleeper reported hearing the sound in a dream; an optional slow mode "
        "rose 0.08 % per cue up to the discrimination threshold and stopped for the night after an arousal."
    ),
    citations=(Citation(arm="Experiment 2", **_KONKOLY_2024),),
    caveats=(
        "A public app: 416 users, 120 analysable; nights with untrained cues or no cues were the controls.",
        "The cue interval and the awakening pause are not restated for Experiment 2; Experiment 1's are used.",
    ),
    cues=(CueSpec("tlr", "two distinct harp melodies of about 3 s, fading in and out", duration_seconds=3.0),),
    timing=(CueTiming("tlr", interval_seconds=(20.0, 40.0), note="as in Experiment 1"),),
    pauses=PauseRules(
        pause_seconds=(("movement", 60.0),),
        pause_after_awakening_seconds=2700.0,
        note="Movement paused cueing for 1 min; the slow mode halted it for the night after an arousal.",
    ),
    ramp=IntensityRamp(
        start="the perceptual discrimination threshold",
        rises=True,
        step_percent_per_cue=0.16,
        cap="the level at which the sleeper reported the sound in a dream; slow mode 0.08 % per cue up to the threshold",
    ),
    training="As in Experiment 1, every night before sleep.",
    runner=RunnerSettings(tlr_interval_seconds=30.0, tlr_tone_duration_seconds=3.0),
    deviations=_deviations(
        PlatformDeviation(
            parameter="TLR cue",
            published="harp melodies of about 3 s with fades",
            remind="one generated 396 Hz tone of 3 s with the player's 0.25 s fades",
            reason="a generated_tone cue has one frequency; the paper gives none",
        ),
        PlatformDeviation(
            parameter="when cues start",
            published="6 h after training, whatever the stage",
            remind="only in gate-open REM periods",
            reason="REMind is REM-gated",
        ),
        PlatformDeviation(
            parameter="awakening pause",
            published="45 min after a reported cue-caused awakening",
            remind="not applied during the night: awakening markers are logged, not read by the scheduler",
            reason="the fidelity report checks the markers against the next cue",
        ),
        _NO_RAMP,
        _LEVEL_IS_GAIN,
    ),
)

KONKOLY2026_PUZZLE = ProtocolPreset(
    name="konkoly2026_puzzle",
    version=1,
    title="Parameters from Konkoly et al. 2026 (puzzle cueing in REM with TLR)",
    runnable=True,
    design_note=(
        "Four unsolved puzzles a night with a 15 s soundtrack each, two of them cued in REM after about 10 TLR "
        "cues, cues every 15 to 30 s from the lowest detectable volume, and in the morning a puzzle-sound "
        "association test before a 4 min retest. REMind runs the puzzle split, the TLR block and the cue "
        "interval; the morning test is muse-tmr record-cue-recall."
    ),
    citations=(KONKOLY_2026,),
    caveats=(
        "Lab nights with PSG; 19 of the 20 participants reported a lucid dream at least once a month, and the "
        "experimenters tailored cueing to each sleeper.",
        "The paper found no overall benefit of the cues on solving.",
    ),
    cues=(
        CueSpec(
            "puzzle",
            "a 15 s soundtrack per puzzle (8 to 12 s of sound, 3 to 7 s of silence) and a spoken title",
            duration_seconds=15.0,
        ),
        CueSpec(
            "tlr",
            "a sequence of 3 pure-tone beeps rising in pitch",
            duration_seconds=0.65,
        ),
    ),
    timing=(
        CueTiming(
            "puzzle",
            interval_seconds=(15.0, 30.0),
            note="blocks of up to 30 cues per puzzle; on average 9.52 music and 2.04 title cues per REM period",
        ),
        CueTiming(
            "tlr",
            interval_seconds=(15.0, 30.0),
            note="about 10 on entering REM and again before switching puzzles; on average 13.3 per REM period",
        ),
    ),
    pauses=PauseRules(
        pause_until_clear=("emg", "alpha", "loss_of_rem_eye_movements", "loss_of_theta"),
        note="Cues were paused and the intensity lowered on any sign of arousal.",
    ),
    ramp=IntensityRamp(
        start="the lowest detectable volume",
        rises=True,
        on_arousal="paused and lowered",
        cap="raised only while the sleeper stayed soundly asleep",
    ),
    training=(
        "In the evening each puzzle for 3 min with its title and soundtrack looping, then every soundtrack "
        "4 times with the puzzle named and feedback; TLR training of about 20 min at 4 am."
    ),
    morning=(
        "A puzzle-sound association test, then 4 min per puzzle with yes or no feedback; puzzles whose sound "
        "was not recognised were left out. A follow-up attempt about 10 days later."
    ),
    runner=RunnerSettings(
        puzzle_count=4,
        cued_puzzle_count=2,
        puzzle_cue_interval_seconds=30.0,
        tlr_repetitions=10,
        tlr_interval_seconds=30.0,
        tlr_tone_frequency_hz=600.0,
        tlr_tone_duration_seconds=0.65,
    ),
    deviations=_deviations(
        _ONE_CUE_PER_PUZZLE,
        _GENERATED_TONES,
        PlatformDeviation(
            parameter="TLR cue",
            published="3 rising beeps, about 650 ms",
            remind="one generated 600 Hz tone of 0.65 s",
            reason="a generated_tone cue has one frequency",
        ),
        PlatformDeviation(
            parameter="signals",
            published="lucid eye signals and sniffs to report working on a puzzle",
            remind="none",
            reason="no EOG and no breathing sensor in the protocol",
        ),
        _NO_RAMP,
        _LEVEL_IS_GAIN,
        _GATE_REM,
    ),
)

BORGHESE2022_REM_HOME = ProtocolPreset(
    name="borghese2022_rem_home",
    version=1,
    title="Parameters from Borghese et al. 2022 (home REM cueing with a headband)",
    runnable=True,
    design_note=(
        "Home REM cueing with automatic staging: once REM had lasted more than 5 min, the sound every 10 s; "
        "interrupted by a new stage (restart after 5 min of REM), by movement (90 s), alpha (45 s), a blink "
        "(10 s) or a poor signal (4 s). REMind keeps the 5 min of REM through the gate's stable window."
    ),
    citations=(BORGHESE_2022,),
    caveats=(
        "Patients with social anxiety disorder; the sound was paired with the positive feedback of exposure "
        "therapy, and a control group heard the same sound in REM without the pairing.",
        "The volume was chosen by each participant in the app.",
    ),
    cues=(CueSpec("puzzle", "a 1 s piano chord paired with the positive feedback of an exposure session", duration_seconds=1.0),),
    timing=(CueTiming("puzzle", interval_seconds=(10.0, 10.0), note="every 10 s while REM lasted"),),
    pauses=PauseRules(
        start_after_rem_seconds=300.0,
        resume_after_rem_seconds=300.0,
        pause_seconds=(("movement", 90.0), ("alpha", 45.0), ("blink", 10.0), ("poor_signal", 4.0)),
        note="A new stage stopped the sounds until 5 min of REM again.",
    ),
    ramp=IntensityRamp(start="chosen by the participant in the app"),
    training="The sound every 10 s during the 3 min of individual positive feedback after each exposure session.",
    runner=RunnerSettings(gate_min_stable_seconds=300.0, puzzle_cue_interval_seconds=10.0),
    deviations=_deviations(
        _ONE_CUE_PER_PUZZLE,
        PlatformDeviation(
            parameter="cue interval at the default stride",
            published="every 10 s",
            remind="at most one puzzle cue per epoch decision, so every 30 s at the default stride",
            reason="run with --stride-seconds 10 to come closer",
        ),
        PlatformDeviation(
            parameter="blink and poor signal pauses",
            published="10 s after a blink, 4 s after a poor signal",
            remind="no blink rule; a poor signal lowers the volume or, repeated, pauses 120 s",
            reason="the arousal guard's artifact rules",
        ),
        PlatformDeviation(
            parameter="movement and alpha pauses",
            published="90 s and 45 s",
            remind="120 s",
            reason="the arousal guard pauses 120 s, longer than published",
        ),
        _LEVEL_IS_GAIN,
        _GATE_REM,
    ),
)

DRESLER2024_MULTISENSORY = ProtocolPreset(
    name="dresler2024_multisensory",
    version=1,
    title="Parameters from Jafarzadeh Esfahani, Picard-Deland et al. 2024 (multisensory REM cueing), auditory channel only",
    runnable=True,
    design_note=(
        "Lab naps with a ZMax headband: cues about 20 s after the first rapid eye movement with low EMG, visual, "
        "auditory and tactile in turn about every 20 s, each cycle 5 dBA louder while REM went on without arousal; "
        "a micro-arousal halted cueing until it passed, then cueing went on one step lower. On a Muse headband "
        "only the auditory cue is left."
    ),
    citations=(ESFAHANI_2024,),
    caveats=(
        "A preprint, not peer reviewed; version 1 was read.",
        "Lab morning naps; cues in one nap and sham in the other.",
    ),
    cues=(CueSpec("tlr", "visual, auditory and tactile cues paired with SSILD training; only the auditory one here"),),
    timing=(
        CueTiming(
            "tlr",
            interval_seconds=(20.0, 20.0),
            note="cues in visual, auditory, tactile order about every 20 s",
        ),
    ),
    pauses=PauseRules(
        start_after_rem_seconds=20.0,
        pause_until_clear=("micro_arousal",),
        note="Halted on a micro-arousal (EMG or alpha rise) and resumed once it had passed.",
    ),
    ramp=IntensityRamp(
        start="the lowest level judged not arousing, from 40 dBA down in 5 dBA steps",
        rises=True,
        step_db_per_cycle=5.0,
        on_arousal="resume one step lower",
        cap="kept after a lucid eye signal",
        background_noise="white noise about 35 dBA",
    ),
    training="About 30 min of senses-initiated lucid dreaming (SSILD) training with the three cues before the nap.",
    runner=RunnerSettings(tlr_interval_seconds=20.0),
    deviations=_deviations(
        PlatformDeviation(
            parameter="modalities",
            published="visual, auditory and tactile cues",
            remind="the auditory cue only, a generated 396 Hz tone of 1 s",
            reason="Muse S Athena has no light or vibration output",
        ),
        _FIRST_CUE_LATENCY,
        _TLR_TIMING,
        _NO_RAMP,
        _LEVEL_IS_GAIN,
        _GATE_REM,
    ),
)

SCHWARTZ2022_IRT_REM = ProtocolPreset(
    name="schwartz2022_irt_rem",
    version=1,
    title="Parameters from Schwartz et al. 2022 (imagery rehearsal therapy with REM cueing)",
    runnable=True,
    design_note=(
        "One sound paired with a rehearsed positive version of a nightmare, then played in REM at home for two "
        "weeks by a headband that detected sleep stages; nightmares per week was the outcome named in advance, "
        "with a follow-up at 3 months. Only the design transfers; no timing figure is in the notes."
    ),
    citations=(SCHWARTZ_2022,),
    caveats=(
        "Patients with nightmare disorder.",
        "Notes only: the cue timing, volume and pause rules of the headband are not in the abstract.",
    ),
    cues=(CueSpec("puzzle", "one sound paired with the rehearsed scenario during imagery rehearsal therapy"),),
    timing=(CueTiming("puzzle", note="in REM at home for two weeks; timing not in the notes"),),
    training="Imagery rehearsal therapy with the sound played during the rehearsal.",
    morning="Nightmares per week at two weeks and at 3 months.",
    runner=RunnerSettings(),
    deviations=_deviations(
        PlatformDeviation(
            parameter="timing",
            published="not known from the notes",
            remind="repository defaults",
            reason="the full text was not read",
        ),
        _GATE_REM,
    ),
)

SANDERS2019_PUZZLE_SWS = ProtocolPreset(
    name="sanders2019_puzzle_sws",
    version=1,
    title="Parameters from Sanders et al. 2019 (puzzle cueing in slow-wave sleep), documentation only",
    runnable=False,
    not_runnable_reason=(
        "Its cues target slow-wave sleep (N3); REMind has no NREM gate, and running the puzzle split on the REM "
        "gate would be a different protocol."
    ),
    design_note=(
        "The cued and uncued puzzle design REMind's puzzle protocol follows, and the morning sound-to-puzzle "
        "recall test before the retest. At home a Zeo headband played a puzzle's sound for 3 min once it had "
        "detected slow-wave sleep for 1.5 min within 2 min, then the next one, over pink noise from sleep onset."
    ),
    citations=(SANDERS_2019,),
    caveats=("Slow-wave sleep, not REM.",),
    cues=(
        CueSpec(
            "puzzle",
            "a 15 s clip per puzzle (8 to 12 s of sound, 3 to 7 s of silence) from 42 nonverbal music clips",
            duration_seconds=15.0,
        ),
    ),
    timing=(CueTiming("puzzle", note="each sound for 3 min, then the next"),),
    pauses=PauseRules(stops_on=("not_slow_wave_sleep",), note="The sound stopped at once when another stage was detected."),
    ramp=IntensityRamp(background_noise="pink noise from sleep onset"),
    training=(
        "Each evening puzzles until 6 stayed unsolved, 2 min each with its sound looping, then three rounds of "
        "sound-to-puzzle recall with feedback."
    ),
    morning="Each of the 6 sounds played, the paired puzzle named; then one 4 min attempt per puzzle.",
    runner=RunnerSettings(puzzle_count=6, cued_puzzle_count=3),
)

_PRESETS: Tuple[ProtocolPreset, ...] = (
    REMIND_DEFAULT_V1,
    KONKOLY2021_NORTHWESTERN,
    KONKOLY2021_NIJMEGEN,
    KONKOLY2021_OSNABRUECK,
    KONKOLY2021_PARIS,
    KONKOLY2024_EXP1,
    KONKOLY2024_EXP2,
    KONKOLY2026_PUZZLE,
    BORGHESE2022_REM_HOME,
    DRESLER2024_MULTISENSORY,
    SCHWARTZ2022_IRT_REM,
    SANDERS2019_PUZZLE_SWS,
)
_BY_NAME: Dict[str, ProtocolPreset] = {preset.name: preset for preset in _PRESETS}
if len(_BY_NAME) != len(_PRESETS):
    raise RuntimeError("protocol preset names must be unique")


def list_protocol_presets() -> Tuple[ProtocolPreset, ...]:
    """Every preset, the default first."""
    return _PRESETS


def get_protocol_preset(name: str) -> ProtocolPreset:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown protocol preset {name!r}; known: {', '.join(_BY_NAME)}") from None


def resolve_protocol_preset(name: Optional[str] = None, *, require_runnable: bool = True) -> ProtocolPreset:
    """The preset a runner uses: ``name``, or ``remind_default_v1`` when None.

    A preset that cannot run raises ``ProtocolPresetNotRunnable`` unless
    ``require_runnable`` is False.
    """
    preset = get_protocol_preset(name or DEFAULT_PROTOCOL_PRESET)
    if require_runnable and not preset.runnable:
        raise ProtocolPresetNotRunnable(f"protocol preset {preset.name} cannot run: {preset.not_runnable_reason}")
    return preset

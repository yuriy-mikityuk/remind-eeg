# FAQ

Answers to what has come up in the issues, mostly #108, #112, #160 and #167. Dated 2026-10-23. Figures and their sources are in [`numbers.md`](numbers.md).

## How accurate is the REM detection?

Nobody knows yet for a Muse night. The only Muse figures come from one EEG-only night, `overnight_20260707` on `p21`, scored against GSSC, an automated stager, not against PSG or a human scorer. The `p21` gate thresholds were tuned on that night and reported on the same night, so they are in-sample. Against GSSC the gate reached a binary REM-vs-not κ of 0.30 with precision 0.30 and recall 0.80 ([PR #115](https://github.com/yuriy-mikityuk/remind-eeg/pull/115)). That κ is not comparable with the five-class κ of sleep-staging papers, such as the vendor's 0.76 for Muse-S against PSG.

Against PSG the detector has only been run on another headband. On the open BOAS dataset, a Bitbrain Ikon referenced to the mastoid, with its EEG alone, the gate reached a binary REM κ of 0.155 on 64 nights with a pair fitted on the other 64, and at the `p21` pair, tuned on a Muse night, it hardly opened ([`validation_external.md`](validation_external.md)).

GSSC itself was trained mostly on C3, C4, F3, F4 from NSRR PSG, and there is no published evaluation of it on forehead-only EEG like the Muse AF7/AF8. Earlier versions of the code called the `p21` pair `VALIDATED` and GSSC `frontal-trained`; both words were dropped for that reason. Getting a real reference needs nights scored against PSG or by a person, tracked in [#116](https://github.com/yuriy-mikityuk/remind-eeg/issues/116).

## Why is the gate so strict on `p21`?

Two things happened on the same night ([#112](https://github.com/yuriy-mikityuk/remind-eeg/issues/112)). First the gate never opened at all: `p21` streams no optics, so the missing heart rate capped the gate's confidence at 0.55, below the 0.70 it needed to open. Since #114 the gate knows when a preset has no optics by design. Then, once it could open, it was open for 724 of 960 epochs, about 75 % of the night, because the heuristic detector runs hot (mean `P_REM` 0.82). Raising the thresholds to 0.90/0.85 brought that down to about 36 % ([PR #115](https://github.com/yuriy-mikityuk/remind-eeg/pull/115)). The detector itself is effectively one feature: against GSSC only `low_delta_power` separates REM ([#116](https://github.com/yuriy-mikityuk/remind-eeg/issues/116)). Reweighting it will change the thresholds again.

## Why does the night preset skip heart rate?

Battery ([#108](https://github.com/yuriy-mikityuk/remind-eeg/issues/108)). With the optics on (`p1034`) the headband drained at 23.34 %/h, about 4.3 h on a charge, and the first overnight attempt died after about 2 h 29 min. Without them (`p21`) it drained at 10.01 %/h on a bench test and 10.60 %/h over a full 8 h night. REM is densest late in the night, so a recording that dies at 4 h misses most of it. The cost is that the detector loses its heart-rate features on `p21`.

## How soon after REM starts can a cue play?

About a minute at the earliest, and nothing has been measured on a real night yet ([#167](https://github.com/yuriy-mikityuk/remind-eeg/issues/167)). Epochs are 30 s long and the gate needs 60 s of stable REM confidence before it opens. Since [#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225) the pilot commands feed the gate the time between epoch starts, so with `--stride-seconds 10` the stable window and the cooldown take as long as at 30 s; before, they ran three times too fast. That also means a shorter stride does not bring the first cue earlier. On a synthetic NREM-to-REM stream (synthetic input, not sleep) it came 60 to 85 s after REM onset at a 30 s stride, depending on where the onset fell in the epoch grid, 80 to 85 s at 10 s and 85 s at 5 s: a window that is still partly NREM does not look like REM, and at 30 s the first all-REM window counts 30 s towards the 60 s where at 10 s it counts 10 s. The table and how it was made are in [`pilot3_replay_cue_simulation.md`](pilot3_replay_cue_simulation.md). An earlier first cue would take a different gate, such as a shorter stable window, which is a separate decision.

## Does the eye-movement proxy use AF7 − AF8?

Yes, but it does not work yet ([#167](https://github.com/yuriy-mikityuk/remind-eeg/issues/167), [#116](https://github.com/yuriy-mikityuk/remind-eeg/issues/116)). It takes the mean absolute difference of successive raw AF7 − AF8 samples at 256 Hz, which mostly measures high frequencies (muscle and a 64 Hz line from the optics), while eye movements are slow. Its score is 1.0 in every epoch, so it carries no information. Since [#225](https://github.com/yuriy-mikityuk/remind-eeg/issues/225) its weight in `P_REM` is 0; it is still computed and exported. Its replacement filters AF7 − AF8 to 0.3 to 5 Hz, which also removes the 64 Hz line, and counts deflections against a threshold taken from the last 5 minutes of the same recording (`rem_event_count`, `rem_density_per_min`, see [`data_model.md`](data_model.md)). It is weighted 0 too until its thresholds are fitted on REM-bearing nights.

## Why did the EEG have extra samples and gaps?

BLE sometimes delivers the same packet twice, and the decoder used to turn every copy into four more samples per channel ([#160](https://github.com/yuriy-mikityuk/remind-eeg/issues/160)). On a 23 min run there were 43 repeats, and 888 of 90,011 EEG subpackets (1.0 %) were lost. Since 0.1.0 the decoder drops a subpacket that repeats one of the last 16 with the same TAG and marks lost ones as gaps without filling them in (#182). The same run showed the headband sampling at about 257.2 Hz by the host clock, not exactly 256.

## `discover` finds nothing on macOS

The terminal or app running Python may lack Bluetooth permission, and macOS then stops the process before Python logs anything. [`USAGE.md`](USAGE.md) has the workaround: a virtual environment launched through the framework `Python.app`.

## Why is there no sound off macOS?

Real playback goes through macOS `afplay`, and there is no other real backend. On Linux or Windows, `--backend system` makes `play-test-cue`, `train-tlr-cue`, `run-pilot4-cueing` and `run-pilot5-full-night` exit with code 2 instead of running without sound. `--backend dry-run`, or `--allow-dry-run` to accept the silent fallback, runs them anyway; `audio_playback.jsonl` then records `backend_name` `dry-run` for every request, and the Pilot 4/5 summary has `resolved_audio_backend_name` `dry-run`.

## Where did `configs/` go?

The four YAML files under `configs/` were never loaded by any code and had drifted from it (for example a 0.7 REM threshold against the gate's 0.80/0.70, and 1.5 s fades against 0.25 s). They were deleted; the settings are the dataclass defaults listed in [`protocol.md`](protocol.md) and [`numbers.md`](numbers.md). The protocol was also described as `Konkoly-like`; it is a one-way, REM-timed cue protocol without eye-signal verification, and the papers it draws on are in [`references.md`](references.md).

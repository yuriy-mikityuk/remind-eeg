# References

The papers the protocol and the REM check lean on, with what each one is used for here. The notes behind this list, with the figures checked against the full texts, are in [`research/2026-10-uniqueness/notes/`](research/2026-10-uniqueness/notes/) (`tmr_tlr_science.md`, `verify_tmr_tlr_science.md`, `gap_2_the_repo_s_only_accuracy_figures_live_so.md`). Citing a paper here does not mean REMind reproduces its result: no night with sleep-time audio has been run yet.

## Cueing in REM sleep and lucidity training

- Konkoly KR, Appel K, Chabani E, et al. Real-time dialogue between experimenters and dreamers during REM sleep. *Current Biology* 31:1417–1427 (2021). https://doi.org/10.1016/j.cub.2021.01.026. Lucid dreamers in PSG-verified REM signalled with left-right eye movements and answered questions. REMind has no eye-signal detection and no two-way channel, which is why its protocol is described as one-way.
- Carr M, Konkoly K, Mallett R, Edwards C, Appel K, Blagrove M. Combining presleep cognitive training and REM-sleep stimulation in a laboratory morning nap for lucid dream induction. *Psychology of Consciousness: Theory, Research, and Practice* 10(4):413–430 (online 2020, issue 2023). https://doi.org/10.1037/cns0000227. Pre-sleep cue training followed by the same cue in REM, the pattern the TLR block follows.
- Konkoly KR, et al. Provoking lucid dreams at home with sensory cues paired with pre-sleep cognitive training. *Consciousness and Cognition* 125:103759 (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11542932. A home app without sleep monitoring; its randomized second experiment had no-cue and untrained-cue control nights, and cue-induced awakenings were frequent.
- Konkoly KR, et al. *Neuroscience of Consciousness* 2026(1):niaf067 (2026). https://academic.oup.com/nc/article/2026/1/niaf067/8456489. Puzzle soundtracks played in REM to frequent lucid dreamers in the lab; no overall cueing effect on solving, cues entered dreams.
- Picard-Deland C, Jafarzadeh Esfahani M, et al. Inducing lucid dreaming with multisensory stimulation: a preregistered multi-center study. *bioRxiv* 2024.06.21.600133, v2 (2026), preprint, not peer reviewed. https://www.biorxiv.org/content/10.1101/2024.06.21.600133. Signal-verified lucid dreams did not differ between cueing and sham in lab naps; cues disrupted sleep in about a quarter of cued REM periods.

## Targeted memory reactivation

- Sanders KEG, Osburn S, Paller KA, Beeman M. Targeted memory reactivation during sleep improves next-day problem solving. *Psychological Science* 30(11):1616–1624 (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6843748. Correction: *Psychological Science* 31(8):1048 (2020). The cued and uncued puzzle design with a morning retest; its cues were played in slow-wave sleep, not REM.
- Schwartz S, Clerget A, Perogamvros L. Enhancing imagery rehearsal therapy for nightmares with targeted memory reactivation. *Current Biology* 32:4808–4816 (2022). https://doi.org/10.1016/j.cub.2022.09.032. Home REM cueing with an automatic stage-detecting headband in a patient group.
- Borghese F, et al. *Frontiers in Psychiatry* (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9281560/. Home REM cueing with a headband that paused cues after movement, alpha, blinks and poor signal, comparable to what the arousal guard does.

## Sleep staging used as a reference

- Hanna J, Flöel A. An accessible and versatile deep learning-based sleep stage classifier. *Frontiers in Neuroinformatics* 17:1086634 (2023). https://www.frontiersin.org/articles/10.3389/fninf.2023.1086634/full. GSSC: trained on 3,314 NSRR PSG nights, mostly on C3, C4, F3, F4, and able to stage from one EEG channel. There is no published evaluation on forehead-only EEG such as the Muse AF7/AF8.
- Hanif U, et al. Deciphering insomnia: benchmarking automated sleep staging algorithms for complex sleep disorders. *Journal of Sleep Research* 34(6):e70048 (2025). https://doi.org/10.1111/jsr.70048. GSSC ahead of YASA on clinical PSG (REM F1 0.76 against 0.35).
- Gunter KM, et al. *medRxiv* 2026.07.06.26357378 (2026), preprint, not peer reviewed. https://www.medrxiv.org/content/10.64898/2026.07.06.26357378v1.full. Open stagers on a portable device with one frontal EEG channel plus EOG and EMG; GSSC was the most robust.
- Lanthier M, et al. *Sleep Advances* zpaf089 (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12782022/. The vendor's closed algorithm on Muse-S against PSG: five-class κ 0.76, REM κ 0.85. A different measure from the binary REM-vs-not κ that `scripts/validate_rem_detection.py` reports, and not comparable with it.
- Jafarzadeh Esfahani M, et al. *bioRxiv* 2023.08.18.553744 (2023), preprint. https://www.biorxiv.org/content/10.1101/2023.08.18.553744v1. ZMax staging; an example of how undisclosed training data makes a stager hard to compare. On ZMax's F7 and F8, referenced to Fpz, an automatic detector's REM eye-movement counts correlated r = 0.66 with its counts on PSG EOG and came out about 1.8 times as high on average, with a wide spread; [`synthetic_data.md`](synthetic_data.md) cites it for why the synthetic eye-movement deflections are an assumption.
- Parry YD, Briganti G. *medRxiv* 2026.06.01.26354591 (2026), preprint. https://www.medrxiv.org/content/10.64898/2026.06.01.26354591v1.full. Home forehead EEG staging on 100 nights; a forehead-trained stager could not be evaluated because it would not install.

## The synthetic optics pulse

[`synthetic_data.md`](synthetic_data.md) builds the optics of a synthetic night from these. The widths and levels come from the maintainer's own recordings, not from the papers.

- Allen J. Photoplethysmography and its application in clinical physiological measurement. *Physiological Measurement* 28(3):R1–R39 (2007). https://doi.org/10.1088/0967-3334/28/3/R01. The pulse as a small cardiac part on a large, slowly varying level that breathing, sympathetic activity and thermoregulation move: here a pulse on a DC level with a wandering baseline.
- Elgendi M. On the analysis of fingertip photoplethysmogram signals. *Current Cardiology Reviews* 8(1):14–25 (2012). https://doi.org/10.2174/157340312801215782. The pulse described by its systolic and diastolic waves.
- Tang Q, Chen Z, Ward R, Elgendi M. Synthetic photoplethysmogram generation using two Gaussian functions. *Scientific Reports* 10:13883 (2020). https://doi.org/10.1038/s41598-020-69076-x. Synthetic beats as the sum of a systolic and a diastolic Gaussian wave. The generator makes the systolic wave log-normal and adds a narrow dip at the pulse foot for the sharp light maximum the recordings show.

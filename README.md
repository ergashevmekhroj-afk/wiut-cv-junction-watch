# Junction Watch — WIUT Hackathon 2026, Computer Vision track

Traffic-event detection (Part A) and causal accident anticipation (Part B) for
the fixed CCTV camera over the Tashkent junction in the sample videos.

## Links

* Website: https://mekhroj1-junction-watch.static.hf.space (approach, EDA, annotated samples, report)
* Live demo: https://junction-watch.streamlit.app (upload a clip, get the events back)
* Predictions on the samples: `predictions_samples.json`

## Install and run

```bash
pip install -r requirements.txt          # Python >= 3.10; Linux + NVIDIA GPU (T4) for the official run
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

No internet is needed at run time. The only model file, `weights/yolo11s.pt`
(19 MB, sha256 `85a76fe8…d502d5`), is committed; `weights/download.sh` re-fetches
it if it is ever missing. `run_submission.py` and `evaluate.py` are the
organisers' files, unchanged.

Dev commands:

```bash
python tools/cache_tracks.py --videos samples                 # detector+tracker once, cached in .cache/
python run_submission.py --videos samples --out predictions_samples.json --team junction-watch
python evaluate.py --pred predictions_samples.json --gt labels/ground_truth_dev.json --per-video
python tools/render.py samples/sample_03.mp4 --pred predictions_samples.json --out renders/sample_03.mp4
python tools/build_flow.py --videos samples                   # re-learn scene/flow_prior.npz
```

Live demo (Streamlit Community Cloud, CPU): `streamlit run demo/streamlit_app.py`
(dependencies in `demo/requirements.txt`, system packages in `demo/packages.txt`).

`tools/label.html` is the labelling tool we used to build `labels/ground_truth_dev.json`
(open it in a browser, load a sample video, mark events with the keyboard).

## Approach

```
video ──► register to reference view (SIFT homography on the median background)
      ──► YOLO11s @960 px, every 3rd frame (≈10 fps) + ByteTrack ──► tracks
      │        └─ same decoding pass: read the pedestrian signal lamp ──► signal phase
      ──► trajectories in reference coordinates (smoothed foot point, velocity, heading)
      ──► per-class rules on trajectories + scene map + signal phase
      ──► segment post-processing (union within class, gap merge, blip removal)
```

**Scene map.** `scene/scene.json` holds the junction layout drawn once on
`scene/reference.jpg` (median background of sample_01, 1280×720): the stop
line of the main carriageway, the three pedestrian crossings, the median and
refuge island, the approach lanes, the junction box and the traffic islands.
The camera is fixed within a video, but the samples are framed differently
(samples 03/04 are shifted ~40 px and zoomed out ~2 % relative to 01/02), so each
video is registered to the reference with a SIFT + RANSAC homography computed
on its own median background; all rules work in reference coordinates.

**Signal phase.** The vehicle heads on the gantry face away from the camera.
The pedestrian head on the left pole faces it; its red and green lamps are read
from a small ROI (colour-difference score ~100 lit vs ~0 dark). On the samples
every stop-line crossing of the main carriageway happens while this lamp is
green, so it runs in phase with the main carriageway. Calibration from the
samples: legal clearing traffic crosses up to 3.9 s after the lamp turns red,
and queues move off up to 5 s before it turns green (sample 02), so `red_light`
also needs behavioural evidence (no group departure, other vehicles still waiting).
The camera can drift slowly (sample_01), so the lamp ROI is re-registered during the pass.

**Learned flow prior.** `scene/flow_prior.npz` stores, per 32 px cell, the mean
driving direction of all vehicle tracks in the samples and its coherence. It
defines the legal direction for `wrong_way`.

### Rules (what is rule-based)

| class | rule |
|---|---|
| `red_light` | crosses the main stop line > 4.5 s into red and > 1.5 s before green, at speed, and drives on into the junction |
| `stop_line` | stops past the stop line (stop zone / crossing) during red; ends when the signal turns green |
| `illegal_u_turn` | heading reverses > 150° after starting on the main carriageway, **and** the reversal is made on a pedestrian crossing or across the solid median. Uzbekistan's Rules of the Road (clause 62) prohibit U-turns on pedestrian crossings; a U-turn inside the intersection from the left lane is otherwise legal and no sign at this junction forbids it, so the common U-turn around the median nose is counted as legal (reported in the EDA, not as an event). Fragments of the same vehicle are stitched |
| `failure_to_yield` | vehicle moving through a crossing (front entering to rear leaving) while a pedestrian on that crossing is within ~90 px of its path; conflicts less than 3 s apart form one event |
| `jaywalking` | pedestrian > 18 px inside the carriageway, > 30 px from every crossing, off islands/median, ≥ 1 s; riders (person over a bike/motorbike box, or median speed above 80 px/s) excluded; people jaywalking within 4 s of each other form one event, kept if it lasts ≥ 6 s |
| `stopped_vehicle` | stationary ≥ 10 s on the carriageway, not explained by the signal, a queue, pedestrians on a crossing ahead, or (< 20 s) waiting for a gap in the junction |
| `wrong_way` | moving against a high-coherence cell of the flow prior for ≥ 1.5 s |
| `accident` | footprints overlap (not across the median), impact-like deceleration (≥ 65 % of speed lost, ≥ 90 px/s²), both stationary ≥ 3 s after |
| `near_miss` | hard braking from ≥ 60 px/s to ≤ 35 % toward a road user on a crossing course (TTC < 1 s), no contact; rejected on ID switches / non-rigid boxes |
| `congestion` | ≥ 8 vehicles in the approach with median speed < 12 px/s for ≥ 20 s that persists through a green phase |

Not predicted (removed from `CLASSES`): `illegal_turn`, `solid_line_crossing`,
`road_obstacle`, `fire_smoke`. A class we predict that never occurs in the test
set adds a zero to the macro average, so we only emit classes whose rule we
checked on the samples.

### Part B — risk score

`src/traffic/risk.py`: a second YOLO11s instance (640 px, ≈7.5 fps) with its own
ByteTrack, registered to the reference view from the **first frame only**. For
each pair of road users, danger = closing speed × exp(−TTC) under constant
velocity; crossing conflicts (heading difference ≥ 30° or a pedestrian) weigh 1.0,
same-direction pairs 0.4. The danger of the worst pair is mapped to a probability
through its quantiles in normal traffic on the samples
(`scene/risk_calibration.json`, 8,271 samples): ordinary traffic stays below 0.5,
and only danger beyond the maximum seen in normal traffic raises an alarm. A short causal mean
suppresses one-frame spikes. The estimator never reads the video file and never
uses Part A output.

### What is learned

* YOLO11s detector: COCO-pretrained weights from Ultralytics, used as is (no fine-tuning).
* Flow prior: statistics of our own tracks on the four sample videos.
* Everything else is rule-based, with thresholds calibrated on the sample videos.

## Evaluation on our dev labels

`labels/ground_truth_dev.json` holds our teammates' labels of `sample_01.mp4` and
`sample_02.mp4` (13 events). Scored with the official `evaluate.py`:

| class | F1 (mean of tIoU 0.3/0.5/0.7) | TP/FP/FN @0.5 | note |
|---|---|---|---|
| jaywalking | 0.40 | 2/4/2 | e.g. sample_01 241.9–253.0 s vs label 241.2–253.9 s |
| failure_to_yield | 0.00 | 0/42/2 | labels are one 38–52 s block per video; we report each car–pedestrian conflict |
| congestion | 0.00 | 0/0/4 | our rule ignores ordinary red-light queues |
| illegal_u_turn | 0.00 | 0/1/1 | the labelled U-turn turns inside the junction, legal under clause 62 |
| road_obstacle | 0.00 | 0/0/2 | not predicted |
| stop_line, red_light | 0.00 | 0/6/0, 0/3/0 | not labelled on these videos |

**Score A = 0.04** on the two labelled videos. The disagreements are mostly
definitional (what counts as one event, whether a queue is congestion, whether a
junction U-turn is illegal), so we changed only the minimum length of a jaywalking event (6 s; the labelled
ones last 11–17 s) and fixed two real bugs:

* brisk walkers near the camera were treated as scooter riders and dropped from `jaywalking` (found through the labels);
* the camera in sample_01 pans ~8 px during its first 20 s, so the lamp ROI (registered on the first frame) slid off the lamp and the whole video read as red. The lamp reader now re-registers on single frames at 2, 5, 10, 20, 40 s and then every 60 s (`REREG_AT` in `pipeline.py`); a red halo on a lamp that has just switched off no longer blocks a lit green (`lamp_red_full`).

`predictions_samples.json` is the full GPU run (Colab T4). After the lamp fix, the
events of sample_01 were recomputed from the same cached detections with the
fixed phase reader. For samples 02–04 the camera does not move, and the fixed reader reproduces
the GPU phase (99.8–100 % agreement), so their events are unchanged.

## Data and models

| item | source | licence |
|---|---|---|
| YOLO11s weights | Ultralytics (COCO-pretrained) | AGPL-3.0 |
| ByteTrack (via Ultralytics) | Zhang et al. 2022 | MIT (original), AGPL-3.0 (Ultralytics port) |
| Sample videos | hackathon organisers | competition use only |

No external training data was used.

## Determinism

Seeds are fixed in `solution.py` (Python, NumPy, PyTorch). Detector inference
and ByteTrack are deterministic for a given GPU/driver; the only nondeterminism
is floating-point noise in cuDNN kernels. Frame sampling is by frame index.

## Runtime

Budget per video: Part A + Part B <= 3 x duration, and the harness decodes every
frame again for Part B. Both parts measure the machine's decoding speed: Part A
stops early (and returns the events found so far) if continuing would not leave
1.5 x one full decoding pass plus post-processing time; Part B stops running its
detector once the remaining frames alone would need the rest of the budget. On
a fast machine neither triggers. `predictions_samples.json` is a full run with
`TRAFFIC_NO_DEADLINE=1` (guards off), i.e. what a machine fast enough not to
trigger them produces. Measured on Colab (T4, 2 vCPU), decoding the 4K samples
alone takes ~1.5 x their duration, so there the guards do trigger. With the
guards off, the full run took ≈2.9 x duration for Part A and ≈3.2 x for Part B on
that machine, dominated by 4K decoding on 2 CPU cores. With the guards on (the default), the
total stays inside the 3 x budget by stopping early.


Part A decodes each video once (every frame is grabbed, every 3rd is decoded
and sent to the detector); Part B receives every frame from the harness and
runs its detector on every 4th. `TRAFFIC_CACHE=off` disables the on-disk
detection cache (the cache only speeds up re-runs during development).

## Team

| member | role | who did what |
|---|---|---|
| Mekhroj | Computer-vision lead | Road-analysis pipeline: detection and tracking, scene map and registration, signal-phase reader, event rules, Part B risk score, evaluation |
| Jamshid | Frontend lead | Team website: layout, sample-video results pages, interactive timelines |
| Nizomiddin | Web and demo engineer | Live demo (upload a video, get events back), EDA charts, deployment |

Previous project we are proud of: an AI camera system that takes attendance automatically (Python, face detection on classroom cameras).

AI assistants were used to help write code, the website and this report, as the rules allow.

## Repository

```
solution.py              interface for the harness (Part A + Part B)
run_submission.py        organisers' harness (unchanged)
evaluate.py              organisers' metric (unchanged)
src/traffic/
  perception.py          detector + tracker, frame sampling, cache
  scene.py               scene map, registration (SIFT homography), geometry helpers
  signal.py              signal phase from the pedestrian lamp
  tracks.py              trajectories, rider marking, ID-switch checks, stitching
  flow.py                learned flow prior
  rules/                 one module per rule family
  segments.py            segment post-processing
  pipeline.py            Part A orchestration
  risk.py                Part B causal risk
scene/                   reference view, scene map, flow prior
tools/                   caching, rendering, review sheets, EDA, labelling tool
demo/                    live demo app (Streamlit)
weights/                 yolo11s.pt (+ download.sh)
```

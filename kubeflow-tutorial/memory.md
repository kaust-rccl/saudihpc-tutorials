# memory.md: resume point for the Kubeflow tutorial work

Written 2026-10-23 so the session can be resumed on another workstation (local GPU + Kubernetes with Kubeflow).
To resume with Claude Code: open this repo, then say *"Read kubeflow-tutorial/memory.md and continue from 'Next steps'."*
Treat everything below as **notes from a previous session, not verified facts about your new machine**. Numbers come from CPU runs.

---------------------------------------------------------------------------------------------------

## 1. The project

Build content for a **Kubeflow tutorial** (agenda: `docs/agenda.csv`, 3 h 10 min, five hands-on exercises). The content needs a real-world
food-security story that follows the **data-science model lifecycle** and exercises all five Kubeflow functions:

1. **Interactive computing** for exploration, development and planning (notebook server, T4)
2. **Kubeflow Pipelines**: cleaning, pre-processing, synthetic dataset generation
3. **Katib**: best-configuration search
4. **Distributed training**: scaling / high throughput
5. **KServe**: serve the good model as a service

**Constraints stated by the user**
- Per participant: **1 x T4 (16 GB) interactive**, **2 GPUs for batch runs through Kueue**. (The agenda text says a *shared* 2-GPU pool; the user later said *per participant*. Not reconciled; confirm.)
- Input data arrives **pre-loaded in a PVC, maximum 5 GB**. No downloads during the tutorial.
- Katib trials are 1-GPU slices, scheduled 2 at a time; distributed training is a 2-GPU job.
- The user will run GPU tests on the other workstation. Do not worry about GPU on the original machine (CPU only there).
- Wants the story validated as a **beta in a notebook first**. **Do not map notebooks to the 5 exercises yet.**
- Agenda still describes the old GFSAD narrative ("Mission Briefing"); it will need updating for the final story.

**Working-style preferences observed:** wants honest results including negative ones; likes a recommendation with the decision laid out; sometimes asks for one-line answers; wants the story understood before it is cut into exercises.

---------------------------------------------------------------------------------------------------

## 2. What was tried, in order (and why it changed)

1. **GFSAD1000 (USGS cropland map, Earth Engine)** -> `notebooks/cropwatch_beta.ipynb`, story "Operation CropWatch" (irrigated vs rainfed, Indo-Gangetic Plain train, Sahel held out).
   Worked as a lifecycle story (honest evaluation by 1-degree blocks, gate that caught a model learning single-year weather, serving with an "unfamiliar input" warning).
   **Dropped as the main spine** because: tabular and tiny (full run ~2 min on CPU, no honest case for GPUs/distributed training), weak 1 km modelled labels, one 2010 snapshot (nothing to validate a refresh against), MLP did not beat a random forest.
2. **Reviewed the datasets in the innovatiana article** (https://www.innovatiana.com/en/post/ai-datasets-for-agriculture). Image datasets fit the 5 functions. Chose **PlantVillage** (training) + **RoCoLe** (real-world check).
3. **Story: "Operation CropGuard"** -> `notebooks/cropguard_beta.ipynb` (the current main artifact). See section 3.
4. **Found an augmentation recipe** with a development/locked split of RoCoLe (section 4).
5. **Tested other backbones** with frozen features (section 5). **This weakens the recipe finding**: it is backbone dependent.

---------------------------------------------------------------------------------------------------

## 3. CropGuard story and datasets

**Story:** a farm-advisory app should diagnose a leaf from a phone photo. Lab-trained models look superb on lab photos; do they survive a real field?
Train the **38-class** PlantVillage model (this stays, user decision). Real-world check on RoCoLe coffee photos the model never trains on.
Crops/diseases do not overlap, so RoCoLe can only test **triage: healthy vs sick**, derived as `sick score = 1 - sum(prob of the 12 healthy classes)`
(a *direct* healthy-vs-sick head is trained as a reference).

**PlantVillage** (training). 54,305 colour leaf photos, 256x256, 38 classes, 14 crops (12 healthy classes, 26 diseased classes).
- Source used: Hugging Face `mohanty/PlantVillage` (files: `data.zip` 2.18 GB containing `raw/color` 0.85 GB, `raw/segmented` 0.51 GB, `raw/grayscale` 0.82 GB; `leaf_grouping/leaf-map.json`; `splits/color_{train,test}.txt`).
- **License: HF card says CC BY-SA 3.0**; some mirrors say CC0. Not resolved; check the original (Mendeley) before redistributing anything derived.
- Traps (all shown in the notebook): 75.7% of images have a leaf id (7,286 leaves, ~5.6 photos each); the shipped split leaks (129 leaves on both sides = 8.1% of test images, 10.7% of the images that have a leaf id);
  68% of classes get >=90% of images from one capture source; class sizes 152 to 5,507; Soybean/Blueberry/Raspberry have only a healthy class, Orange/Squash only a disease class.
- Parsing gotchas: some colour filenames have no `___`; segmented files have a different UUID and a `_final_masked.jpg` suffix, so join colour<->segmented by key (the lower-cased name after `___`, extension and `_final_masked` removed).
- Notebook uses a **subset**: 200 images per class, whole leaves (7,629 images, 3,064 leaves), resized to 224, with a leaf mask from the segmented image. Split **by leaf** 60/20/20 (train 4,589 / val 1,483 / test 1,557).

**RoCoLe** (real-world check, never trained on). 1,560 smartphone photos of Robusta coffee leaves, 2048x1152, one plantation in Ecuador (CIIDEA, Calceta), 390 plants x 4 photos.
- Labels: healthy 791, red spider mite 167, rust level 1/2/3/4 = 344/166/62/30 (sick share ~49%).
- **License CC BY 4.0** (confirmed from Mendeley's own API). Paper: Parraga-Alava et al., Data in Brief 2019, doi 10.17632/c5yvn32dzg.2.
- Download that works: `https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2` (2.2 GB zip: 1.57 GB of photos plus a 698 MB VOC archive that is not needed). Labels: `Annotations/RoCoLE-csv.csv`, column `Label` is JSON with `"classification"`. Plant id = filename minus the last 6 characters.
- Notebook prep: centre square crop of each photo, resize to 224.

**Protocol (important):** RoCoLe is split **by plant** (seed 123) into a **development half (195 plants)**, used to choose the augmentation recipe and head settings, and a **locked half (195 plants)**, scored for the final result.
This replaced the original "RoCoLe never influences any choice" rule once the user asked to *find* an augmentation regime. Bootstrap confidence intervals resample plants.
**Caveat:** the locked half has now been scored several times for ResNet-18 + the chosen recipe. If a new backbone/recipe is adopted, choose on the development half only and consider a fresh split or a single final look.

---------------------------------------------------------------------------------------------------

## 4. Results so far (CPU, ResNet-18 frozen features, 3 seeds unless noted)

- Shortcut probe (3,000-image subset, split by leaf; chance 2.6%): background only **67.7%**, leaf only **86.6%**, whole photo **89.2%**. The backdrop leaks the label.
- Random split vs split by leaf (logistic regression): **95.2% vs 91.7%**.
- Harsh synthetic field (earlier approach): clean-trained model drops to **7.7%** accuracy; **not a reliable proxy** for real-field benefit.

**Recipe sweep on the RoCoLe development half** (derived-triage AUC / direct-head AUC / clean PV val accuracy):

| recipe | derived AUC | direct AUC | PV val acc |
|---|---|---|---|
| none | 0.583 | 0.714 | 0.910 |
| synth0.5 / synth1.0 (field synthesis) | 0.665 / 0.628 | 0.686 / 0.718 | 0.870 / 0.861 |
| flip+photo1 | 0.648 | 0.734 | 0.904 |
| flip+rrc0.25 (crop only) | 0.633 | 0.770 | 0.914 |
| flip+rrc0.25+photo1 | 0.688 | 0.775 | 0.900 |
| flip+rrc0.5+photo1 | 0.696 | 0.794 | 0.899 |
| **flip+rrc0.1+photo1 (chosen)** | **0.747** | 0.746 | 0.902 |
| bgswap+rrc0.25+photo1 | 0.671 | 0.746 | 0.879 |
| synth0.35+rrc0.35+photo0.6 | 0.627 | 0.738 | 0.860 |

Recipe notation: `flip` = rot90/mirror, `rrcS` = random resized crop with minimum area S, `photoS` = photometric jitter strength S, `synthS` = field synthesis, `bgswap` = leaf onto another lab backdrop.
Two copies of the augmented set did not help; repeat draws of a recipe move AUC by ~+-0.01; differences under ~0.03 on 195 plants are noise.
Selection rule (fixed before looking at the locked half): highest dev derived AUC among recipes that keep clean val accuracy within 5 points and do not hurt the direct reference.

**Locked half, notebook run (before = clean only, after = chosen recipe, same head settings):**
derived triage AUC **0.573 -> 0.768** (+0.196, 95% CI +0.163 to +0.230); balanced accuracy 0.505 -> 0.625; direct-head reference AUC 0.724 -> 0.754;
PlantVillage clean test accuracy 91.5% -> 90.6%; share of coffee leaves called sick (truth ~49%) 98% -> 76% (still biased toward "sick").
A separate 5-seed scratch run gave a similar gain (+0.176). Illustrative production target (derived AUC >= 0.80) is **not met**; artifact is labelled beta.

Familiarity check (5-NN cosine distance to a reference feature set): flags ~49% of RoCoLe photos with the augmented reference (98% with a clean-only reference), 100% of a small non-leaf smoke set.
Confidence table on clean PV test: coverage/accuracy 0.955/0.927 at p>=0.5, 0.742/0.982 at p>=0.9. The serving demo's first photo (an apple scab leaf) is a **confident miss**; kept on purpose.

---------------------------------------------------------------------------------------------------

## 5. Backbone comparison (the finding that qualifies section 4)

`experiments/backbones.py`, frozen features, linear head, **RoCoLe development half only**, same recipe `flip+rrc0.1+photo1` (chosen on ResNet-18):

| backbone | feat dim | derived AUC before->after | direct AUC before->after | PV val acc before->after |
|---|---|---|---|---|
| ResNet-18 | 512 | 0.571 -> 0.706 | 0.706 -> 0.736 | 0.891 -> 0.893 |
| ResNet-50 (V2 weights) | 2048 | 0.670 -> 0.645 | 0.649 -> 0.731 | 0.901 -> 0.898 |
| EfficientNet-B0 | 1280 | 0.722 -> 0.660 | 0.766 -> 0.758 | 0.924 -> 0.931 |
| ConvNeXt-Tiny | 768 | 0.710 -> 0.749 | 0.895 -> 0.885 | 0.934 -> 0.943 |
| DINOv2-Small (ViT-S/14, via torch.hub) | 384 | **0.809 -> 0.881** | **0.918 -> 0.926** | **0.959 -> 0.965** |

Reading: **the backbone matters more than the augmentation.** A bigger ResNet is not clearly better; better pretrained features (ConvNeXt, DINOv2) are. The augmentation gain seen with ResNet-18 is **largely repair of weak features** and does not
replicate on ResNet-50/EfficientNet. Caveats: recipe tuned on ResNet-18 only, untuned linear heads, one image draw, 195 dev plants, frozen features only (fine-tuning may reorder things).
CPU extraction time for 13.8k images: ResNet-18 164 s, EfficientNet-B0 156 s, ResNet-50 423 s, ConvNeXt-T 585 s, DINOv2-S 591 s.
**Unverified:** DINOv2 weight license (check it); weights download from the internet (must be placed on the PVC for the tutorial).

---------------------------------------------------------------------------------------------------

## 6. Repo map and how to run

```
kubeflow-tutorial/
  memory.md                   this file
  README.md                   quick start
  requirements.txt            pip packages (see GPU note)
  docs/agenda.csv             the tutorial agenda (20 segments; exercise order is Notebooks, Katib, Trainer, KFP, KServe)
  notebooks/cropguard_beta.ipynb   MAIN: PlantVillage + RoCoLe story, 12 sections (outputs stripped)
  notebooks/cropwatch_beta.ipynb   earlier GFSAD story (reference only; needs Earth Engine auth)
  gfsad/                      scripts from the GFSAD exploration
  experiments/                sweep.py, sweep2.py, final_eval.py, backbones.py + the augmentation snippets
```
**Not in the repo (by design):** datasets, caches (`notebooks/data/`), model artifacts (`notebooks/artifacts/`), executed notebook outputs (stripped; all key numbers are in this file).

**To run the main notebook:** `cd kubeflow-tutorial/notebooks`, open `cropguard_beta.ipynb`, run all. If `data/raw/` is empty it downloads PlantVillage (HF) and RoCoLe (Mendeley) itself, ~4.4 GB.
First run builds `data/cache/` (preprocessed arrays and feature files). On the CPU box that was ~10 min for the notebook plus ~15 min for the augmentation/regime feature caches; a GPU extracts faster.
Section 9 (GPU only) fine-tunes ResNet-18 before/after augmentation and scores the locked half; it is skipped without CUDA.
**Section 9 has only been smoke-tested on CPU (2 training steps). The CUDA + mixed-precision path is untested.**
The experiment scripts read the notebook's cache: run them from `kubeflow-tutorial/` with `CROPGUARD_CACHE=notebooks/data/cache` (default) after the notebook has built it.
`EE_PROJECT` (env var) is your Google Cloud project id; only the GFSAD notebook/scripts need it (it was removed from the committed files).

**Environment on the original machine:** conda env `kf-tutorial-26`, Python 3.13.15, kernel name `kf-tutorial-26`.
torch 2.14.1 (**CPU build**), torchvision 0.29.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6, pyarrow 26, matplotlib 3.11.2, seaborn 0.13.2, pillow 13.0.0, jupyter/notebook 7.6.3, nbconvert 7.17.2, earthengine-api 1.7.49.
**On the GPU machine install the CUDA build of torch/torchvision** and register the kernel (`python -m ipykernel install --user --name kf-tutorial-26`), or edit the notebook's kernelspec.

---------------------------------------------------------------------------------------------------

## 7. Corrections to earlier claims (do not repeat the originals)

- "Augmentation costs nothing on clean accuracy": **false**; it costs ~1 to 3 points.
- "Augmentation clearly helps real-world transfer": **only for ResNet-18**; backbone dependent (section 5).
- Earlier direct-head probes (simpler synthetic field + logistic regression) showed a gain that the notebook's harsh synthetic regime did **not**; the effect depends on the recipe and the backbone.
- An early statement said 10.7% of test images leak; the notebook's figure is **8.1%** of all test images (10.7% is of those with a leaf id).
- "RoCoLe never influences any choice": superseded by the dev/locked split (section 3).
- The derived 38-class triage score ranks real photos **worse than a direct healthy-vs-sick head** unless a good recipe/backbone is used.

---------------------------------------------------------------------------------------------------

## 8. Next steps (suggested order)

1. **Run `cropguard_beta.ipynb` on the GPU machine**, including section 9. Check the CUDA/AMP path works; record fine-tuned before/after numbers on the locked half.
2. **Add a backbone option to the notebook** (ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, DINOv2-S). Choose **backbone + recipe jointly on the development half**; score the locked half once. Re-check the recipe per backbone (it was tuned on ResNet-18).
3. **Fine-tune** the leading backbones on GPU (the real question for the tutorial: does a fine-tuned network still need this augmentation, and what does 2-GPU DDP buy?). Measure images/s and time-to-accuracy for 1 vs 2 GPUs.
4. **Resolve open items:** PlantVillage license (CC BY-SA 3.0 vs CC0), DINOv2 license, 1 shared vs 2 per-participant GPU quota, whether the repo is public (outputs were stripped and the GCP project id moved to an env var as a precaution).
5. **Only then** design the Kubeflow pieces (nothing is built yet: no YAML, no pipeline, no InferenceService). Candidate ideas, not agreed with the user:
   - *KFP:* prepare (decode/resize/mask/dedupe, shard) -> augment -> extract/train -> evaluate -> gate; RoCoLe resize/align is a natural pre-processing job.
   - *Katib:* search backbone, crop min-area, photometric strength, aug fraction, lr, weight decay; objective = RoCoLe **development** AUC with a clean-accuracy floor; 1 GPU per trial, 2 in parallel.
   - *Trainer:* 2-GPU DDP fine-tune with the winning settings.
   - *KServe:* the `CropGuardService` in the notebook (photo in; diagnosis or "refer to extension officer" out) as a custom predictor.
6. Update `docs/agenda.csv` text for the CropGuard narrative, and decide the mapping to exercises last.

---------------------------------------------------------------------------------------------------

## 9. Gotchas learned the hard way

- `pkill -f "<pattern>"` in a Bash command kills your own shell if the command line contains the pattern. Use PID files (`echo $! > pid; kill -0 $PID`).
- `np.savez(path, **d)` fails if `d` has a key named `file`; the RoCoLe cache uses `filename`.
- `torch.load(..., weights_only=True)` needs plain Python numbers (not numpy scalars) inside saved dicts.
- Matplotlib 3.11 renamed `boxplot(labels=)` to `tick_labels=`.
- Mendeley's "files" API returns an empty list; use the zip URL. HF `mohanty/PlantVillage` is gated by nothing but is large; `data.zip` holds all three image variants.
- Earth Engine pixel pulls: `ee.data.computePixels` with a 256x256 tile on the GFSAD grid works (~8 s per 128x128 tile with 37 bands); needs `ee.Initialize(project=...)`.
- `jupyter nbconvert --execute` writes output only at the end; wait on the PID, not on output files.

---------------------------------------------------------------------------------------------------

## 10. Session 2 (2026-10-23, GPU workstation): Exercise 1 student notebook and fine-tuned ConvNeXt-Tiny

**Machine.** DGX Spark (`dgx-spark-01`, aarch64, **one NVIDIA GB10**, 121 GB unified memory, driver 580, CUDA 13) running a single-node Kubernetes with Kubeflow installed (CRDs for Notebooks, Katib Experiments, Trainer ClusterTrainingRuntimes, KServe; **Kueue not checked**).
conda env `kf-tutorial-26` created with `torch 2.14.1+cu130` / `torchvision 0.29.1+cu130` from `https://download.pytorch.org/whl/cu130` (aarch64 wheels exist); kernel registered. This is **not** a T4 and has one GPU, so 2-GPU DDP and "2 trials at a time" cannot be tested here.

**Built.**
- `data_prep/build_cropguard_dataset.py` -> `data/cropguard/` (1.55 GB, ~75 s from the raw archives; matches the beta: 7,629 photos, split 4,589/1,483/1,557 by leaf, 390 plants = 195 dev + 195 locked). Layout: `MANIFEST.json`, `plantvillage/{images.npy,masks.npy,meta.csv}`, `rocole/{images.npy,meta.csv}`, `weights/convnext_tiny_imagenet1k_v1.pth`. Splits are stored in the meta tables (`split`, `half`).
- `notebooks/exercise1_workspace_and_baseline.ipynb`: GPU/pod check, data volume (PVC), ConvNeXt-Tiny fine-tune on lab photos (baseline), field test on RoCoLe, backdrop probe, `flip+rrc0.1+photo1` augmentation fix, bootstrap, scorecard, save, reflection. fp16 autocast + GradScaler (a T4 has no bf16), channels-last, data-loader workers sized from the cgroup CPU quota and `/dev/shm`.
  Full run **3 min 49 s on the GB10** (baseline 65 s, augmented 119 s, ~320 photos/s, peak 4.2-4.4 GiB at batch 64). **Untested on a T4 and inside a pod.** The PVC mount path (`/data/cropguard`) is an assumption.

**Results, ConvNeXt-Tiny fine-tuned 4 epochs, 6 seeds (42, 7, 123, 1, 2, 3).** Triage AUC on RoCoLe, derived from the 38-class output:
- Lab test accuracy **97.3%** baseline vs 97.2% with the recipe (no cost).
- Field AUC, locked half: baseline **0.848 +- 0.059** (range 0.754-0.917), with recipe **0.880 +- 0.041** (0.803-0.918). Paired gain **+0.032 +- 0.032 sd (SE 0.013)**, positive in 6/6 seeds. Dev half: baseline 0.816, with recipe 0.854, gain **+0.038**, positive in 4/6.
  So the recipe helps a little on a fine-tuned ConvNeXt (vs **+0.2** on frozen ResNet-18), and **one seed cannot see it**: the baseline's own seed-to-seed spread (~0.06 AUC, range 0.16) is larger than the effect. Seed 42 (the notebook default) shows only +0.008.
- Fine-tuning the whole network lifts the field ranking a lot vs the frozen features in section 5 (dev AUC ~0.82 vs 0.71 frozen ConvNeXt-T and 0.57 frozen ResNet-18).
- **The default 0.5 cut-off is useless**: ~97% of coffee leaves are called sick, balanced accuracy 0.52 (baseline) / 0.53 (recipe), because coffee is outside the 14 trained crops. A cut-off **tuned on the dev half** gives locked balanced accuracy 0.78 (baseline) / 0.81 (recipe).
- Backdrop probe on the baseline (test photos): with the leaf painted out 14-27% accuracy (chance 2.6%); leaf only 80-93%; whole photo ~97%.
- The bootstrap over plants only covers plant sampling noise, **not** training-seed noise.

**Locked half is now heavily used** (6 seeds x 2 ConvNeXt-Tiny models, on top of the earlier ResNet-18 scoring). Treat it as consumed for ConvNeXt-Tiny + this recipe; make any further model/recipe choices on the dev half and consider a fresh plant split for a final look.

**Implications for the next steps.**
- **Katib (exercise 2): a trial's objective (dev AUC of one training run) has run-to-run sd ~0.05**, about the size of the effects being searched. A naive 1-run objective would mostly select noise: average replicates, search coarser knobs, or use a more stable metric. Decide how to teach this.
- Exercise 1 currently shows the recipe as the "manual fix". If instructors want Katib to discover the augmentation settings, hide step 8 or keep it as a stretch.
- 15-minute lab budget: the whole notebook takes ~4 min on the GB10; confirm on a T4.
- Notebook source was authored in `# %%` percent format and converted; the `.ipynb` is now the source of truth (outputs stripped).

**Exercise 1 notebook frozen (2026-10-24).** `notebooks/exercise1_workspace_and_baseline.ipynb`, sha256 `649b2f3faaa22a8d17884782783bcce4fa488218626b2e011656f8b0eba58e28`.
Verified by a full run on the GB10: 29/29 code cells, 0 errors, 66 s + 120 s training; baseline lab accuracy 97.0%; field ranking AUC 0.846 dev / 0.862 locked; tuned cut-off 0.73 gives 0.79 on locked. Outputs stripped.
Text was shortened and made plain (no ResNet or earlier-test references); results table, cut-off curve and "knobs cell" callouts added. `exercise1_workspace_and_baseline_GB10.ipynb` is the earlier executed copy (older wording and table layout) and does not match the frozen text.
**Still untested:** a T4, and the save/run-log cells inside a pod. The "Reference run" numbers in the notebook are from the GB10 (seed 42); swap in T4 numbers after a T4 dry run.

---------------------------------------------------------------------------------------------------

## 11. Exercise 2 (Katib): plan and pilot (started 2026-10-24)

**Real cluster (stated by the user):** Kueue installed, **8 T4 GPUs for batch shared by 20 participants**. This dev box (dgx-spark-01, one GB10) has Katib v0.19.0 (random, grid, hyperopt/TPE, skopt/Bayesian, Optuna, Hyperband; StdOut metrics collector; namespace `mshaikh` has metrics injection on) but **no Kueue** and one GPU. Exercise 3 as planned (a 2-GPU job each) needs 40 GPUs for 20 people: only 4 jobs fit at once, so it needs its own plan.

**Budget logic.** 8 GPUs x 15 min / 20 people = **360 GPU-seconds each**: 3 to 4 cheap trials, not 12. Class wall time (20 people, 8 GPUs, perfect packing, T4 assumed 2x slower than GB10 = unmeasured): 12 trials x 4 epochs ~70 min; 12 x 2 epochs ~43 min; 4 x 2 epochs ~14 to 19 min.

**Decision (user):** 4 trials, 2 epochs, **one dial only: learning rate, 4 values** (3e-5, 1e-4, 3e-4, 1e-3). Augmentation fixed (off in the pilot). Objective: dev ranking score (AUC on the dev half of RoCoLe); safety check lab accuracy >= 0.95; never touch the locked half. Plan: the instructors pre-run a bigger search to show next to the students' mini-search; Exercise 3 retrains the winner at full length as the "did it win again?" check.
Open: augmentation fixed on or off in the lab; how to deliver the trial container (zot registry vs stock notebook image + script); T4 timing.

**Built.** `exercise2/train_trial.py` (Exercise 1's training as a script; prints `dev_auc=` and `val_acc=` for Katib's StdOut collector; no augmentation, never scores locked). `exercise2/pilot_lr.py` (resumable runner: one process per trial, results appended to `exercise2/pilot/pilot_lr_results.jsonl`). `exercise2/summarize_pilot.py` (table of results).

**Pilot, first 8 trials (seeds 11, 12; GB10; 2 epochs; dev AUC):**
| lr | seed 11 | seed 12 | lab acc |
|---|---|---|---|
| 3e-5 | 0.799 | 0.726 | 0.76 (under-trained: fails the 0.95 check) |
| 1e-4 | 0.875 | 0.794 | 0.95 |
| 3e-4 | 0.867 | 0.841 | 0.97 |
| 1e-3 | 0.819 | 0.756 | 0.97 |
Shape: a hill. Both extremes are worse than the middle two in both seeds; 1e-4 vs 3e-4 is a toss-up (the seed-to-seed gap at the same lr is 0.03 to 0.08, bigger than the 0.02 between them). **Cost: ~43 s per trial on the GB10** (35 s training), so a T4 is likely 1.5 to 2 min per trial including pod start-up. Unmeasured.

**Background run in progress (started 2026-10-24):** seeds 13..22 x 4 learning rates = 40 more trials, ~29 min, PID in `exercise2/pilot/pilot.pid`, log `exercise2/pilot/pilot.log`.
Resume: `cd kubeflow-tutorial`; check `kill -0 $(cat exercise2/pilot/pilot.pid) && echo running`; `tail -3 exercise2/pilot/pilot.log` (ends with `PILOT_DONE`); `python exercise2/summarize_pilot.py`. If it died, rerun `python exercise2/pilot_lr.py --seeds 13 14 15 16 17 18 19 20 21 22` (it skips finished trials).
**Next:** read the 12-seed table (does the hill hold? which lr does a single 4-trial search pick how often?); decide augmentation on/off and the lr grid; check zot pull from the cluster and the `mshaikh` data volume (the `data`/`gpu-check-datavol-1` PVCs were RWX, which `local-path` cannot provide); run one trial as a Kubernetes Job; then write the Experiment YAML and the student notebook.

**Pilot finished (12 seeds x 4 learning rates = 48 trials, 0 failures, ~44 s per trial on the GB10, 2 epochs, dev half only):**
| lr | dev AUC mean +- sd | range | lab accuracy |
|---|---|---|---|
| 3e-5 | 0.720 +- 0.081 | 0.59 to 0.83 | 0.755 (fails the 0.95 check) |
| 1e-4 | 0.835 +- 0.041 | 0.77 to 0.88 | 0.943 (just under the check) |
| **3e-4** | **0.845 +- 0.028** | 0.79 to 0.90 | **0.975** |
| 1e-3 | 0.753 +- 0.059 | 0.66 to 0.85 | 0.973 |
Within-seed winners (what one 4-trial search would pick): 3e-4 in 6/12 seeds, 1e-4 in 5/12, 1e-3 in 1/12, 3e-5 never. Best single trial: lr 3e-4, seed 16, dev AUC 0.895, lab 98.2% (a lucky draw, not the expectation).
Reading: learning rate matters at the extremes (too low under-trains, too high is worse by ~0.09); **1e-4 vs 3e-4 is a toss-up** (0.010 apart, noise ~0.03 per trial). 3e-4 is the pick on lab accuracy and spread. Two epochs at 3e-4 (dev AUC 0.845) is no worse than Exercise 1's four epochs at 2e-4 (0.816 +- 0.065 over 6 seeds), at half the cost.
Results: `exercise2/pilot/pilot_lr_results.jsonl`; summary: `python exercise2/summarize_pilot.py`.

**Frozen-stages test (2026-10-24; lr 3e-4, 2 epochs, no augmentation, dev half; seeds 11 and 12; GB10).** `train_trial.py --freeze-stages N` keeps the stem and the first N stages (N = 1..3) at pretrained weights; 4 = train only the new classifier. N = 0 reproduced the earlier pilot exactly (0.8671, 0.8409). Results: `exercise2/pilot/pilot_freeze_results.jsonl`.
| frozen | dev AUC (seed 11 / 12) | mean | lab acc | seconds per trial |
|---|---|---|---|---|
| none | 0.867 / 0.841 | 0.854 | 0.974 | 43 |
| stem + stage 1 | 0.881 / 0.853 | 0.867 | 0.976 | 35 (-20%) |
| up to stage 2 | 0.881 / 0.865 | 0.873 | 0.972 | 30 (-32%) |
| up to stage 3 | 0.850 / 0.786 | 0.818 | 0.944 (fails 0.95) | 23 (-48%) |
| everything (head only) | 0.754 / 0.712 | 0.733 | 0.687 | 20 (-55%) |
Reading: a plateau then a cliff. Freezing the first one or two stages is free (+0.01 to +0.02, within the ~0.03 noise) and saves 20 to 32% of the time; freezing three stages starts to hurt and breaks the lab-accuracy check; head-only collapses at this lr and length (lr likely interacts: a head-only run would want a higher lr). Two seeds only, lr fixed, GB10 timings.

**Drop-path test (2026-10-24; lr 3e-4, 2 epochs, nothing frozen, no augmentation, dev half; seeds 11 and 12; GB10).** `train_trial.py --drop-path P` (stochastic depth, torchvision default 0.1; default left unchanged, 0.1 reproduced 0.867 / 0.841 exactly). Results: `exercise2/pilot/pilot_droppath_results.jsonl`.
| drop-path | dev AUC (seed 11 / 12) | mean | lab acc | change vs 0.1 (per seed) |
|---|---|---|---|---|
| 0.0 | 0.871 / 0.850 | 0.860 | 0.979 | +0.004 / +0.009 |
| 0.1 (default) | 0.867 / 0.841 | 0.854 | 0.974 | - |
| 0.2 | 0.878 / 0.835 | 0.857 | 0.973 | +0.011 / -0.006 |
| 0.4 | 0.893 / 0.867 | 0.880 | 0.966 | +0.026 / +0.026 |
Reading: a mild dial. 0.4 is best in both seeds (+0.026, about one run-to-run sd, so suggestive not proven) but lab accuracy slips about 1 point (still above 0.95); 0.0 to 0.2 are indistinguishable. No cost change. Whether more than 0.4 helps (or lab accuracy keeps falling) is untested. Two seeds only.

**Same accuracy, faster: time cap + early stopping (2026-10-24; lr 3e-4, 2 epochs; GB10).** Objective set by the user: keep 2 epochs, match the 3e-4 run's accuracy in no more than its time. `train_trial.py` gained `--max-seconds`, `--stop-rule`, `--stop-auc`, `--curve`, `--benchmark` (see its docstring); with no options it reproduces the old results exactly (dev AUC 0.8671).
1. **Free speed-up: cuDNN auto-tuning off.** It cost ~8 s at the start of every trial (clock read 17 s at 25% of training but 4 to 7 s per later quarter). Off: train 35.1 s -> 30.5 s, whole trial 43.5 s -> 34.6 s (-20%), results identical (0.8671 / 0.8409). It buys ~15% steady-state speed, so keep it ON only for long runs (Exercise 3). Now the trial default. Untested on a T4.
2. **Calibration curves** (`exercise2/pilot/curve_results.jsonl`, seeds 11 and 12): lab accuracy on 500 held-out leaves at 50% of training: 3e-5 0.64, 1e-4 0.90 to 0.91, 3e-4 0.95 to 0.97, 1e-3 0.91 to 0.92 (ends 0.97 anyway). Lab accuracy alone cannot catch lr 1e-3 (its fault is the field score): dev ranking score at 50%: 1e-3 0.72 to 0.75, 3e-4 0.82 to 0.89.
3. **Rule:** at 50% of training stop if lab accuracy < 0.88 or dev score < 0.78. Cap: 33.5 s (= 1.1 x the 3e-4 run's 30.5 s).
4. **Validated on fresh seeds 13 and 14** (`exercise2/pilot/early_stop_results.jsonl`): 3 of 8 trials stopped (3e-5 both seeds, 1e-3 seed 13), all three would have ended badly in the full runs; 1e-4 and 3e-4 ran to the end with results identical to the full runs (e.g. 3e-4: 0.826 vs 0.824, 0.858 vs 0.858); 1e-3 seed 14 was not stopped (final 0.804, vs 0.717 in its earlier full run: high lr is chaotic, tiny numerical changes move it by ~0.09). **8 trials took 256 s vs 351 s (-27%)**; a 4-trial search per person: -22% to -31%. The cap never triggered here (runs ended at ~31 s); it is a safety net, and on a T4 it must be recalibrated from that machine's reference run.
Thresholds were set on seeds 11/12 and checked on 13/14 (8 trials: small sample). Not yet combined with `--freeze-stages 2` or image size.

**One-trial Kubernetes Job prepared, not yet run (2026-10-24).** `exercise2/k8s/trial-job.yaml` + `run_trial.sh` (from `kubeflow-tutorial/`: `bash exercise2/k8s/run_trial.sh`). Namespace `mshaikh`; image = the notebook image `jupyter-pytorch-cuda-full:v1.11.0` (already on the node); script delivered by ConfigMap `cropguard-train-trial` (10 KB, from `train_trial.py`); data from PVC `test-datavol-1` mounted read-only at `/data1`, dataset at `/data1/data/cropguard` (MANIFEST verified; the `gpu-check-datavol-1` PVC is still Pending because it was created RWX); 1 GPU, 4 to 8 CPUs, 2 Gi `/dev/shm`; `sidecar.istio.io/inject: "false"` (Istio is on in the namespace; a sidecar would stop the Job from ever finishing; Katib trial templates need the same). Server-side dry run passes (pod security `baseline` accepts it).
**Blocker:** notebook `mshaikh/test-0` holds the node's only GPU, so the Job would stay Pending. Stopping it (annotation `kubeflow-resource-stopped`) frees the GPU for a ~1 min trial; it is the user's notebook, so ask first. Expected result: same numbers as the GB10 runs for seed 11 (dev AUC ~0.867, lab acc ~0.975).

**One-trial Job RAN on the cluster (2026-10-24, after the user freed the GPU by removing `test-0`).** `bash exercise2/k8s/run_trial.sh`: Job `cropguard-trial-demo` Complete in 43 s (pod start to finish), single container (no Istio sidecar), GPU visible, data read from the PVC. Printed: train_seconds=31.4, total_seconds=36.5, peak 4.2 GiB, early_stopped=0, **val_acc=0.9791, dev_auc=0.8684** (GB10 outside Kubernetes: 0.9757 / 0.8671; the difference is run-to-run numerical noise). At the 50% checkpoint: lab 0.962, dev score 0.879 (passes both stop rules). The Job deletes itself after 1 h; ConfigMap `cropguard-train-trial` stays. Still a GB10, not a T4.
**Next:** write the Katib Experiment YAML around this same pod spec (parameter `lr` in {3e-5, 1e-4, 3e-4, 1e-3}, objective `dev_auc` maximise, additional metric `val_acc`, StdOut collector, `parallelTrialCount` 1 here), run it, then the student notebook.

**Exercise 1 notebook verified out of the box in a pod (2026-10-24).** Mount decision: PVC `test-datavol-1` has the dataset at `<volume>/data/cropguard`. The Jobs mount it with `subPath: data` at `/data`, so the dataset is at **`/data/cropguard`**, the notebook's second search location (after `$CROPGUARD_DATA`). `exercise2/k8s/notebook-job.yaml` ran the frozen notebook unchanged (no env var, no edits) via nbconvert: **29/29 cells, 0 errors, 4 min**; same numbers as the GB10 (baseline 66 s, augmented 122 s; ranking 0.85 dev / 0.86 locked; tuned cut-off 0.73 -> 0.79 locked; bootstrap lines identical); the save step and run log also worked in the pod (`/home/jovyan/cropguard-artifacts/exercise1/`, inside the container, so ephemeral). The trial Job was re-run with the new mount (dev AUC 0.8682, lab 0.9791).
**Caveat for notebook servers made in the Kubeflow UI:** the form has no subPath, so mounting `test-datavol-1` at `/data` would give `/data/data/cropguard` (not found). Either move the dataset up one level inside the volume (`<volume>/cropguard`, then mount at `/data`; not done, it is the user's volume) or set `CROPGUARD_DATA` (a PodDefault is cleanest).

**FIRST KATIB EXPERIMENT RAN (2026-10-24; GB10 cluster, namespace `mshaikh`).** `exercise2/k8s/katib-experiment.yaml`: Experiment `cropguard-lr-search`, grid over lr {3e-5, 1e-4, 3e-4, 1e-3} (categorical list), objective maximise `dev_auc`, extra metric `val_acc`, StdOut collector, parallel 1, same seed 11, 2 epochs, time cap 33.5 s + early-stop rules inside the script. Submit: `kubectl apply -f exercise2/k8s/katib-experiment.yaml`; inspect: `kubectl get experiment,trials -n mshaikh`. Succeeded: 4/4 trials, **3 min 7 s end to end**, no failures.
| lr | dev_auc | lab acc | trial seconds |
|---|---|---|---|
| 3e-5 | 0.7511 | 0.6285 | 29 (stopped early at 50%) |
| **1e-4 (Katib's pick)** | **0.8737** | 0.9514 | 45 |
| 3e-4 | 0.8681 | 0.9791 | 45 |
| 1e-3 | 0.8621 | 0.9703 | 45 |
Reading: Katib chose 1e-4, but it beats 3e-4 by 0.006, far inside the ~0.03 run-to-run noise, and its lab accuracy (0.9514) only just clears our 0.95 safety check. The earlier 12-seed pilot says 1e-4 and 3e-4 are a toss-up and prefers 3e-4 on lab accuracy. **Katib only optimises the objective it is given; our lab-accuracy guard is not enforced** (fold it into the metric, or read `val_acc` by hand). The early stop worked (3e-5 took 29 s vs 45 s); 3e-4 reproduced the standalone Job (0.8681 vs 0.8682). 1e-3 behaved this time (0.862; it was 0.82 to 0.72 in other runs: chaotic).
Notes: Katib deletes finished trial pods (no logs afterwards; metrics stay in the Katib DB); set `trialTemplate.retain: true` to keep pods if the lab should show worker logs. The metrics collector rewrites the trial's stdout into its own file, so `kubectl logs -c trial` is empty while a trial runs. Experiment and trial objects were left in the namespace.

**Katib round 2: Bayesian search + Katib median-stop (2026-10-24; GB10 cluster).** Files: `exercise2/k8s/katib-bayes-medianstop.yaml` (experiment `cropguard-bo-lr`), `katib-earlystop-check.yaml` (control `cropguard-es-check`). Trial script gained `--log10-lr` (unused now), `--report` (prints `dev_auc=` and `val_acc=` at the `--curve` checkpoints, which Katib's median-stop needs).
- **Bayesian optimisation works.** Settings: `random_state 7`, `n_initial_points 3` (default 10 would make 8 trials pure random), seed 11, 2 epochs, parallel 1, 8 trials in ~7 min. **Katib's skopt service samples every double parameter log-uniformly** (`Real(min, max, "log-uniform")`, undocumented; the range must be positive). My first attempt searched log10(lr) in [-4.5, -3.0], got `nan` and failed. Results: the 3 random starts were 3e-5 (0.799), 9.6e-4 (0.850), 2.3e-4 (0.869); it then probed 3.2e-4 (0.862), **1.41e-4 (0.8814, lab 0.9717: best)**, 9.2e-5 (0.873, lab 0.948), 1.23e-4 (0.878), 5.7e-4 (0.854). It concentrated on 9e-5 to 3e-4, the good zone found by hand. Best-vs-neighbours gaps are inside the ~0.03 noise.
- **Median-stop stops trials, but in this Katib (v0.19.0) the stopped trial gets NO observation and no completion time.** Seen 3 times: a trial below the median at 50% of training is killed correctly (e.g. lr 3e-5, running average ~0.73 vs median 0.86), but then the experiment controller holds back new suggestions (`requests = len(trials) + add - incompleteEarlyStopping`), so **an experiment that still needs more trials hangs** (BO run 1: stuck at 4 of 8 for 10+ min). Not caused by `retain` (tried both), nor by the missing `val_acc` at stop time (now reported at each checkpoint; the stopped trial still has no observation). Probable cause: no completion time/zero-timestamp metric logs; unconfirmed. In the control (`es-check`, 5 trials) the stopped trial was the last one, so the experiment finished: 4 Succeeded + 1 EarlyStopped.
- **Median-stop rule semantics:** needs `min_trials_required` finished trials first (so the first 3 trials are never stopped); threshold = median over finished trials of the AVERAGE of all values each reported (a bad trial among them drags the median down: 0.78 instead of 0.84); stops a trial whose running average is LESS after `start_step` reports. Our in-script rule (calibrated, validated on fresh seeds) does not have these problems.
- **Katib grid visits points in random order** (Optuna sampler), so "good then bad" cannot be arranged.
- **PBT: blocked here, and a poor fit.** `katib-config` gives the PBT suggestion a **ReadWriteMany** 5 Gi volume; the only storage class (`local-path`) is ReadWriteOnce, so the volume would never bind (the cluster-wide fix is a change to `katib-config` in `kubeflow`: not done, needs the user's approval). Needs `resumePolicy: FromVolume`, `suggestion_trial_dir`, `n_population` (example uses 40), `truncation_threshold`, and a trial script that loads its parent's checkpoint and saves a new one every generation (the exact checkpoint contract is only in Katib's `simple-pbt` image, not seen). With 2-epoch trials PBT means many tiny generations, each paying ~15 s of pod start-up: more total GPU time than the Bayesian run, so it does not fit the 8-T4 / 20-person budget. Not run.
Experiments left in `mshaikh`: `cropguard-lr-search` (grid), `cropguard-bo-lr` (Bayesian, finished), `cropguard-es-check` (control).

**Katib round 3: Hyperband, all brackets (started 2026-10-24 ~18:12 local; GB10).** `exercise2/k8s/katib-hyperband.yaml`, experiment `cropguard-hb`: algorithm hyperband, `resource_name=epochs`, `eta=2`, `r_l=4` (rungs of 1, 2, 4 epochs; ~14 to 15 trials, ~34 epochs, about 12 min on the GB10), `parallelTrialCount: 4` (Katib requires >= ceil(eta^s_max); there is ONE GPU, so 3 of 4 trial pods wait), `maxTrialCount: 15`, lr searched as `log10_lr` in [-4.5, -3.0] (assumed uniform sampling => log scale; if the first trials fail with nan, the sampler is log-uniform: pass `lr` directly instead), seed 11, no early stopping (Hyperband halves the field itself). Progress log: `exercise2/pilot/hyperband.log` (watcher pid in `exercise2/pilot/hyperband.pid`, ends with `WATCH_DONE` and a per-trial table).
Reconnect: `tail -5 exercise2/pilot/hyperband.log`; `python exercise2/summarize_katib.py cropguard-hb` (works any time, also for the other experiments); `kubectl get experiment,trials -n mshaikh`.
Facts from the source (release-0.19): the resource must be an int parameter listed in `parameters`; the service restarts survivors from scratch with a bigger resource (no checkpoint resume), so a trial at 4 epochs repeats the work of its 1- and 2-epoch versions; `parallelTrialCount` below ceil(eta^s_max) is rejected; a stopped trial's metrics are not an issue here because Hyperband does not use early-stopping.

**Hyperband RESULT: stalls after the first rung in Katib v0.19.0 (checked 2026-10-24 21:11 local, experiment started 18:12).** 4 of ~15 trials done, then nothing for 3 hours. Cause (controller log): `The response contains unexpected trials`. The experiment controller asks the suggestion for `parallelTrialCount` (4) new trials; Hyperband replies with only the survivors of the rung (2), the controller rejects the mismatch and retries forever. `parallelTrialCount` must be >= ceil(eta^s_max) (validated), so the bracket sizes (4, 2, 1, 3, 2, 3) can never match a fixed request size: **a full multi-rung Hyperband cannot run in this Katib build** (the upstream example only runs one rung: eta 2, r_l 2, 2 trials). Not fixable without a different Katib version or a custom suggestion. The experiment `cropguard-hb` is left in this stuck state in `mshaikh` (delete: `kubectl delete experiment cropguard-hb -n mshaikh`).
**What the one rung that ran shows (1 epoch each, seed 11):** lr 9.6e-4: dev 0.853, lab 0.958; 8.5e-4: 0.851, 0.969; 9.1e-5: 0.816, lab 0.819; 3.4e-5: 0.713, lab 0.47. At a 1-epoch budget the highest learning rates win (they learn fastest), so **a short first rung would promote 1e-3 and eliminate 1e-4**, while at 2 epochs 1e-4 to 3e-4 are the good ones (pilot: 0.835 to 0.845 vs 0.753 for 1e-3). Low-budget ranking is a biased proxy for this problem: a second reason to avoid Hyperband here.

**STABLE Katib experiment (2026-10-24):** `exercise2/k8s/katib-ui-bayes.yaml`, run from the Katib UI as `new-bo-ui` (spec identical to the file, checked field by field): Bayesian, `random_state 7`, `n_initial_points 3`, 4 trials, parallel 1, our in-script early stopping, **Succeeded in 3 min 17 s**. Trials: lr 9.6e-4 dev 0.8877 / lab 0.9724 (Katib's best); 2.3e-4 0.8697 / 0.9751; 1.0e-4 0.8737 / 0.9528; 6.0e-4 0.8644 / 0.9791. Differences between them are inside the ~0.03 run-to-run noise. Exercise 1 is pushed (commit e483967 "stable version" = origin/dev, notebook identical to the working copy). Exercise 2 files committed after this.
UI lessons: the form's trial-parameter rows come from the `${trialParameters.x}` placeholders in the template box (the MNIST example brings `momentum`); each needs a Reference to a hyperparameter; "Edit and submit YAML" needs the WHOLE Experiment, the template box only the Job; no Save button, the final CREATE button sends everything; validator errors appear in `kubectl logs deploy/katib-ui -n kubeflow -c katib-ui`.

**What is in git for Exercise 2 (decision by the user, 2026-10-24):** only `exercise2/k8s/katib-ui-bayes.yaml` is committed. Everything else under `exercise2/` (train_trial.py, the pilot and summary scripts, the other experiment YAMLs, `exercise2/README.md`, `exercise2/pilot/*.jsonl`) stays LOCAL on the workstation and is not in the repo, although the notes above describe it. The stable YAML alone is not runnable from a fresh clone: it needs the ConfigMap `cropguard-train-trial` built from `train_trial.py`. Back up `exercise2/` before wiping the workstation.

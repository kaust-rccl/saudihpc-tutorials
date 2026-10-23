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
  docs/agenda.csv             the tutorial agenda (19 segments)
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

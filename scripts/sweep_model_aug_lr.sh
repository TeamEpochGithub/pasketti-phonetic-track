#!/usr/bin/env bash
# =============================================================================
# sweep_model_aug_lr.sh — 20-run hyperparameter sweep
# =============================================================================
#
# Sweep design (20 runs total):
#   Group A (12 runs): 4 models × 3 augmentation levels  @ mid LR
#   Group B  (8 runs): 4 models × 2 extra LR combos      @ aug=low
#
# Fixed hyperparameters:
#   LoRA rank     = 16
#   LoRA alpha    = 32
#   LoRA dropout  = 0.05
#   Classifier dropout = 0.1
#   Batch size    = 16   (decrease if OOM; increase to 32 if VRAM allows)
#   Mixed precision = true
#   Early stopping patience = 5
#   Epochs = 20
#
# Models (all ~90M params):
#   1. facebook/wav2vec2-base
#   2. microsoft/wavlm-base
#   3. microsoft/wavlm-base-plus
#   4. facebook/hubert-base-ls960
#
# Augmentation levels:
#   none — all augmentations disabled  (augmentation=dummy)
#   low  — gentle time-stretch + pitch-shift only
#   high — aggressive time-stretch + pitch-shift + band-stop filter
#
# Learning-rate combos:
#   mid  — head_lr=1e-3,  lora_lr=8e-5
#   low  — head_lr=2e-4,  lora_lr=2e-5
#   high — head_lr=4e-3,  lora_lr=1.5e-4
#
# Usage:
#   chmod +x scripts/sweep_model_aug_lr.sh
#   ./scripts/sweep_model_aug_lr.sh          # run ALL 20 sequentially
#   ./scripts/sweep_model_aug_lr.sh 5        # run only command #5
#   ./scripts/sweep_model_aug_lr.sh 13 20    # run commands #13 through #20
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Common fixed overrides ─────────────────────────────────────────────────
COMMON=(
  model.lora_r=16
  model.lora_alpha=32
  model.lora_dropout=0.05
  model.classifier_dropout=0.1
  training.dataloader.batch_size=32
  training.mixed_precision=true
  training.early_stopping_patience=5
  training.num_epochs=20
  logging.project_name=model_aug_lr_sweep
  cv.holdout=true # set to false to do 5-fold CV instead of holdout test set
)

# ─── Augmentation presets ────────────────────────────────────────────────────
AUG_NONE=(
  augmentation.time_stretch.enabled=false
  augmentation.pitch_shift.enabled=false
  augmentation.band_stop_filter.enabled=false
)

AUG_LOW=(
  augmentation.time_stretch.enabled=true
  augmentation.time_stretch.min_rate=0.92
  augmentation.time_stretch.max_rate=1.08
  augmentation.time_stretch.p=0.3
  augmentation.pitch_shift.enabled=true
  augmentation.pitch_shift.min_semitones=-2.0
  augmentation.pitch_shift.max_semitones=1.5
  augmentation.pitch_shift.p=0.3
  augmentation.band_stop_filter.enabled=false
)

AUG_HIGH=(
  augmentation.time_stretch.enabled=true
  augmentation.time_stretch.min_rate=0.8
  augmentation.time_stretch.max_rate=1.2
  augmentation.time_stretch.p=0.6
  augmentation.pitch_shift.enabled=true
  augmentation.pitch_shift.min_semitones=-4.0
  augmentation.pitch_shift.max_semitones=3.0
  augmentation.pitch_shift.p=0.6
  augmentation.band_stop_filter.enabled=true
  augmentation.band_stop_filter.min_center_freq=200.0
  augmentation.band_stop_filter.max_center_freq=4000.0
  augmentation.band_stop_filter.min_bandwidth_fraction=0.5
  augmentation.band_stop_filter.max_bandwidth_fraction=1.99
  augmentation.band_stop_filter.p=0.4
)

# ─── Helper ──────────────────────────────────────────────────────────────────
RUN_NUM=0

run_experiment() {
  local desc="$1"; shift
  RUN_NUM=$((RUN_NUM + 1))

  # Range filtering
  if [[ -n "${START:-}" ]] && (( RUN_NUM < START )); then return; fi
  if [[ -n "${END:-}"   ]] && (( RUN_NUM > END   )); then return; fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  Run $RUN_NUM / 20 — $desc"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  uv run python src/run_pipeline.py "$@"
}

# Parse optional range args
START="${1:-}"
END="${2:-}"
if [[ -n "$START" ]] && [[ -z "$END" ]]; then END="$START"; fi #hihihihi

# =============================================================================
# GROUP A: Model × Augmentation sweep  (12 runs, LR = mid)
# =============================================================================
LR_MID=(training.head_lr=2e-3 training.lora_lr=16e-5) # for 32 batch size 

# ── wav2vec2-base ────────────────────────────────────────────────────────────
run_experiment "wav2vec2-base | aug=none | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/wav2vec2-base \
  "${AUG_NONE[@]}" "${LR_MID[@]}"

run_experiment "wav2vec2-base | aug=low | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/wav2vec2-base \
  "${AUG_LOW[@]}" "${LR_MID[@]}"

run_experiment "wav2vec2-base | aug=high | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/wav2vec2-base \
  "${AUG_HIGH[@]}" "${LR_MID[@]}"

# ── wavlm-base ──────────────────────────────────────────────────────────────
run_experiment "wavlm-base | aug=none | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base \
  "${AUG_NONE[@]}" "${LR_MID[@]}"

run_experiment "wavlm-base | aug=low | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base \
  "${AUG_LOW[@]}" "${LR_MID[@]}"

run_experiment "wavlm-base | aug=high | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base \
  "${AUG_HIGH[@]}" "${LR_MID[@]}"

# ── wavlm-base-plus ─────────────────────────────────────────────────────────
run_experiment "wavlm-base-plus | aug=none | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base-plus \
  "${AUG_NONE[@]}" "${LR_MID[@]}"

run_experiment "wavlm-base-plus | aug=low | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base-plus \
  "${AUG_LOW[@]}" "${LR_MID[@]}"

run_experiment "wavlm-base-plus | aug=high | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base-plus \
  "${AUG_HIGH[@]}" "${LR_MID[@]}"

# ── hubert-base-ls960 ───────────────────────────────────────────────────────
run_experiment "hubert-base-ls960 | aug=none | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/hubert-base-ls960 \
  "${AUG_NONE[@]}" "${LR_MID[@]}"

run_experiment "hubert-base-ls960 | aug=low | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/hubert-base-ls960 \
  "${AUG_LOW[@]}" "${LR_MID[@]}"

run_experiment "hubert-base-ls960 | aug=high | lr=mid" \
  "${COMMON[@]}" model.pretrained_name=facebook/hubert-base-ls960 \
  "${AUG_HIGH[@]}" "${LR_MID[@]}"

# =============================================================================
# GROUP B: Learning-rate sweep  (8 runs, aug = low)
# =============================================================================
LR_LOW=(training.head_lr=4e-4  training.lora_lr=4e-5) #already 2x these
LR_HIGH=(training.head_lr=8e-3 training.lora_lr=3e-4)

# ── wav2vec2-base ────────────────────────────────────────────────────────────
run_experiment "wav2vec2-base | aug=low | lr=low" \
  "${COMMON[@]}" model.pretrained_name=facebook/wav2vec2-base \
  "${AUG_LOW[@]}" "${LR_LOW[@]}"

run_experiment "wav2vec2-base | aug=low | lr=high" \
  "${COMMON[@]}" model.pretrained_name=facebook/wav2vec2-base \
  "${AUG_LOW[@]}" "${LR_HIGH[@]}"

# ── wavlm-base ──────────────────────────────────────────────────────────────
run_experiment "wavlm-base | aug=low | lr=low" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base \
  "${AUG_LOW[@]}" "${LR_LOW[@]}"

run_experiment "wavlm-base | aug=low | lr=high" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base \
  "${AUG_LOW[@]}" "${LR_HIGH[@]}"

# ── wavlm-base-plus ─────────────────────────────────────────────────────────
run_experiment "wavlm-base-plus | aug=low | lr=low" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base-plus \
  "${AUG_LOW[@]}" "${LR_LOW[@]}"

run_experiment "wavlm-base-plus | aug=low | lr=high" \
  "${COMMON[@]}" model.pretrained_name=microsoft/wavlm-base-plus \
  "${AUG_LOW[@]}" "${LR_HIGH[@]}"

# ── hubert-base-ls960 ───────────────────────────────────────────────────────
run_experiment "hubert-base-ls960 | aug=low | lr=low" \
  "${COMMON[@]}" model.pretrained_name=facebook/hubert-base-ls960 \
  "${AUG_LOW[@]}" "${LR_LOW[@]}"

run_experiment "hubert-base-ls960 | aug=low | lr=high" \
  "${COMMON[@]}" model.pretrained_name=facebook/hubert-base-ls960 \
  "${AUG_LOW[@]}" "${LR_HIGH[@]}"

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Sweep complete!  $RUN_NUM / 20 runs executed."
echo "  Check W&B project: model_aug_lr_sweep"
echo "════════════════════════════════════════════════════════════════"

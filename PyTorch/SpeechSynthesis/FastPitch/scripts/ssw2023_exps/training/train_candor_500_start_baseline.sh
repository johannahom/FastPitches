#!/usr/bin/env bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda-11.0.1_460.32/targets/x86_64-linux/lib
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=32}
: ${GRAD_ACCUMULATION:=1}
: ${OUTPUT_DIR:="/disk/scratch3/jomahony/ssw2023_models/fine-tune-ljBase500-candor-ssw-baseline"}
: ${DATASET_PATH:=None}
: ${TRAIN_FILELIST:=filelists/ssw2023/candor_baseline/ssw_baseline_train.csv}
: ${VAL_FILELIST:=filelists/ssw2023/candor_baseline/ssw_baseline_val.csv}
: ${AMP:=false}
: ${SEED:=""}

: ${LEARNING_RATE:=0.1}

# Adjust these when the amount of data changes
: ${EPOCHS:=1000}
: ${EPOCHS_PER_CHECKPOINT:=50}
: ${WARMUP_STEPS:=1000}
: ${KL_LOSS_WARMUP:=100}

# Train a mixed phoneme/grapheme model
: ${PHONE:=true}
# Enable energy conditioning
: ${ENERGY:=true}
: ${TEXT_CLEANERS:=english_cleaners_v2}
# Add dummy space prefix/suffix is audio is not precisely trimmed
: ${APPEND_SPACES:=false}

#Using durations from textgrids
: ${INPUT_TYPE:=transcription}
: ${DURATION_EXTRACTION_METHOD:=textgrid}

#Loading preextracted features
: ${LOAD_DURATION_FROM_DISK:=true}
: ${LOAD_PITCH_FROM_DISK:=true}
: ${LOAD_MEL_FROM_DISK:=true}

# For multispeaker models, add speaker ID = {0, 1, ...} as the last filelist column
: ${NSPEAKERS:=500}
: ${SAMPLING_RATE:=22050}

# For adding a new discrete conditioner.
: ${NCONDITIONS:=3}

# For adding experiment names to wandb
: ${PROJECT:=fine-tune-ljBase500-candor-ssw-baseline}
: ${EXPERIMENT_DESC:=baseline model with fine-tuning from epoch 500 ssw}

# Pitch normalisation
# Whole corpus
#: ${CORPUSMEAN:=160.39}
#: ${CORPUSSTD:=54.61}
# Per speaker
: ${NORM_BY_SPEAKER:=true}


# Adjust env variables to maintain the global batch size: NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 256.
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

ARGS=""
ARGS+=" --cuda"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --dataset-path $DATASET_PATH"
ARGS+=" --training-files $TRAIN_FILELIST"
ARGS+=" --validation-files $VAL_FILELIST"
ARGS+=" -bs $BATCH_SIZE"
ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
ARGS+=" --optimizer lamb"
ARGS+=" --epochs $EPOCHS"
ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"
ARGS+=" --resume"
ARGS+=" --warmup-steps $WARMUP_STEPS"
ARGS+=" -lr $LEARNING_RATE"
ARGS+=" --weight-decay 1e-6"
ARGS+=" --grad-clip-thresh 1000.0"
ARGS+=" --dur-predictor-loss-scale 0.1"
ARGS+=" --pitch-predictor-loss-scale 0.1"

# Autoalign & new features
ARGS+=" --kl-loss-start-epoch 0"
ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"
ARGS+=" --text-cleaners $TEXT_CLEANERS"
ARGS+=" --n-speakers $NSPEAKERS"
ARGS+=" --n-conditions $NCONDITIONS"
ARGS+=" --input-type $INPUT_TYPE"
ARGS+=" --duration-extraction-method $DURATION_EXTRACTION_METHOD"
#ARGS+=" --pitch-mean $CORPUSMEAN"
#ARGS+=" --pitch-std $CORPUSSTD"

[ "$PROJECT" != "" ]               && ARGS+=" --project \"${PROJECT}\""
[ "$EXPERIMENT_DESC" != "" ]       && ARGS+=" --experiment-desc \"${EXPERIMENT_DESC}\""
[ "$AMP" = "true" ]                && ARGS+=" --amp"
[ "$PHONE" = "true" ]              && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]             && ARGS+=" --energy-conditioning"
[ "$SEED" != "" ]                  && ARGS+=" --seed $SEED"
[ "$LOAD_MEL_FROM_DISK" = true ]   && ARGS+=" --load-mel-from-disk"
[ "$LOAD_PITCH_FROM_DISK" = true ] && ARGS+=" --load-pitch-from-disk"
[ "$LOAD_DURATION_FROM_DISK" = true ] && ARGS+=" --load-duration-from-disk"
[ "$NORM_BY_SPEAKER" = true ] && ARGS+=" --norm-pitch-by-speaker"
[ "$PITCH_ONLINE_DIR" != "" ]      && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch
[ "$PITCH_ONLINE_METHOD" != "" ]   && ARGS+=" --pitch-online-method $PITCH_ONLINE_METHOD"
[ "$APPEND_SPACES" = true ]        && ARGS+=" --prepend-space-to-text"
[ "$APPEND_SPACES" = true ]        && ARGS+=" --append-space-to-text"


if [ "$SAMPLING_RATE" == "44100" ]; then
  ARGS+=" --sampling-rate 44100"
  ARGS+=" --filter-length 2048"
  ARGS+=" --hop-length 512"
  ARGS+=" --win-length 2048"
  ARGS+=" --mel-fmin 0.0"
  ARGS+=" --mel-fmax 22050.0"

elif [ "$SAMPLING_RATE" != "22050" ]; then
  echo "Unknown sampling rate $SAMPLING_RATE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

#: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
python train.py $ARGS "$@"
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
: ${WAVEGLOW:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="/disk/scratch3/jomahony/interspeech2023_models/turn_model/FastPitch_checkpoint_900.pt"}
: ${BATCH_SIZE:=20}
: ${PHRASES:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/scripts/inference_test.tsv"}
: ${OUTPUT_DIR:="./test_inference/audio_$(basename ${PHRASES} .tsv)"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=119}
: ${NUM_SPEAKERS:=447}
: ${CONDITION:=0}
: ${NUM_CONDITION:=3}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --waveglow $WAVEGLOW"
ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --repeats $REPEATS"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --speaker $SPEAKER"
ARGS+=" --n-speakers $NUM_SPEAKERS"
ARGS+=" --condition $CONDITION"
ARGS+=" --n-conditions $NUM_CONDITION"
[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

mkdir -p "$OUTPUT_DIR"

python inference.py $ARGS "$@"

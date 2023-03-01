#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
: ${HIFIGAN:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/g_02500000"}
: ${HIFIGAN_CONFIG:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/config.json"}
: ${FASTPITCH:="/disk/scratch3/jomahony/interspeech2023_models/FastPitch_checkpoint_500.pt"}
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

: ${SPEAKER:=77}
: ${NUM_SPEAKERS:=447}
: ${CONDITION:=2}
: ${NUM_CONDITION:=3}

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" -i $PHRASES"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --fastpitch $FASTPITCH"
ARGS+=" --hifigan $HIFIGAN"
ARGS+=" --hifigan-config $HIFIGAN_CONFIG"
#ARGS+=" --wn-channels 256"
ARGS+=" --batch-size $BATCH_SIZE"
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

python inference_hifi.py $ARGS "$@"

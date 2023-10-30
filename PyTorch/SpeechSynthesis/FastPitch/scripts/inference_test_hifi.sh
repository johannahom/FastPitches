#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
: ${HIFIGAN:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/g_02500000"}
: ${HIFIGAN_CONFIG:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/config.json"}
#: ${FASTPITCH:="/disk/scratch3/jomahony/ssw2023_models/fine-tune-ljBase200-candor-ssw-baseline/FastPitch_checkpoint_700.pt"}
: ${FASTPITCH:="/disk/scratch3/jomahony/ssw2023_models/fine-tune-ljBase200-candor-ssw-condition/FastPitch_checkpoint_1000.pt"}
: ${PHRASES:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/scripts/inference_test_candor.tsv"}
: ${BATCH_SIZE:=20}
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

: ${SPEAKER:=64}
: ${NUM_SPEAKERS:=500}
: ${CONDITION:=2}
: ${NUM_CONDITION:=3}
: ${DURATION_EXTRACTION:=textgrid}
: ${DENOISER_STRENGTH:=0.01}

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
ARGS+=" --duration-extraction-method $DURATION_EXTRACTION"
ARGS+=" --denoising-strength $DENOISER_STRENGTH"

[ "$CPU" = false ]          && ARGS+=" --cuda"
[ "$CPU" = false ]          && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]           && ARGS+=" --amp"
[ "$PHONE" = "true" ]       && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]      && ARGS+=" --energy-conditioning"
[ "$TORCHSCRIPT" = "true" ] && ARGS+=" --torchscript"

mkdir -p "$OUTPUT_DIR"

python inference_hifi.py $ARGS "$@"

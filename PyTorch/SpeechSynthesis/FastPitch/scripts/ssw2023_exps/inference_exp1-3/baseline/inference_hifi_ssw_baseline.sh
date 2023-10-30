#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
: ${HIFIGAN:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/g_02500000"}
: ${HIFIGAN_CONFIG:="/disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/pretrained_models/hifigan/config.json"}
: ${FASTPITCH:="/disk/scratch3/jomahony/ssw2023_models/fine-tune-ljBase200-candor-ssw-baseline/FastPitch_checkpoint_1000.pt"}
: ${PHRASES:="/disk/scratch3/jomahony/ssw2023_model_inference/exp1-3_inference/candor_data/inference_fastpitch/filelists/inference_${1}_${2}_baseline.tsv"}
: ${BATCH_SIZE:=20}
: ${OUTPUT_DIR:="/disk/scratch3/jomahony/ssw2023_model_inference/exp1-3_inference/inference_ssw_exps_output/${1}/${2}/baseline/"}
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${PHONE:=true}
: ${ENERGY:=true}
: ${DENOISING:=0.01}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}

: ${SPEAKER:=${2}}
: ${NUM_SPEAKERS:=500}
: ${CONDITION:=0} #where 0 is no label, 1 is turn-medial and 2 is turn-final
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

python /disk/scratch/s2132904/interspeech_2023/FastPitches/PyTorch/SpeechSynthesis/FastPitch/inference_hifi.py $ARGS 
#"$@"

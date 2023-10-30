source /disk/scratch/s2132904/miniconda/bin/activate
conda activate fastP_venv


## declare chosen speakers
declare -a spkr=(1 21 22 33 37 42 44 48 52 53 63 76 93 101 106 132 140 143 145 150 156 176 200 211)

for k in "${spkr[@]}"
do
  ./scripts/ssw2023_exps/inference_mos_speaker_test/inference_hifi_ssw_baseline.sh 1000 ${k} 0.01
done





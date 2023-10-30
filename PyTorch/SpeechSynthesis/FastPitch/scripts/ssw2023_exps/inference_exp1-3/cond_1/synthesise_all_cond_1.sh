source /disk/scratch/s2132904/miniconda/bin/activate
conda activate fastP_venv


## declare chosen speakers and systems
declare -a spkr=(176 48 200 143 156)
declare -a text_type=(final medial ambiguous)

for s in "${spkr[@]}"
do
  for t in "${text_type[@]}"
  do
    ./scripts/ssw2023_exps/inference_exp1-3/cond_1/inference_hifi_ssw_cond_1.sh ${t} ${s}

  done
done






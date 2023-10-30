source /disk/scratch/s2132904/miniconda/bin/activate
conda activate fastP_venv


## declare parameter arrays
declare -a dn=("0" "0.01")
#declare -a ckpt=("300" "400" "500" "600" "700" "800" "900" "1000")
declare -a spkr=(1 20 21 22 24 25 28 33 37 42 44 48 52 53 55 58 63 64 70 76 78 80 87 89 93 98 101 106 108 132 133 140 143 145 150 151 154 156 158 166 168 171 176 181 182 183 187 194 198 200 205 210 211)
## now loop through the above array
for i in "${dn[@]}"
do
#  for j in "${ckpt[@]}"
  for j in {300..1000..100}
  do
    for k in "${spkr[@]}"
    do
      ./scripts/ssw2023_exps/inference_pre_mos/inference_hifi_ssw_baseline.sh ${j} ${k} ${i}
    done
  done
done





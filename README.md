create the conda environment
``` bash
conda env create -f environment.yml
conda activate rf_hourglass
```

install natten for neighborhood attention
``` bash
pip install natten==0.21.5+torch290cu128 -f https://whl.natten.org
```

### ACDC
download & preprocess the dataset
``` bash
./scripts/download_acdc.sh
```

run training on the ACDC dataset

``` bash
python train_acdc.py --config configs/config_acdc_v3_noLerp.json --name Run_100_noLerp --batch-size 32 --grad-accum-steps 1 --max-epochs 1000 --wandb-project AVSS2026-acdc --sample-steps 1 --evaluate-every 20000 --demo-every 20000 --save-every 5000 --compile --num-workers 16
```

### brats 2020
download the dataset
``` bash
python ./utils/get_brats2020.py
```

run training on the brats 2021 dataset

``` bash
python train_brats20.py --config configs/config_mmDiT_brats20.json --name Run_1 --batch-size 32 --max-epochs 300 --wandb-project AVSS-brats2020 --sample-steps 1 --evaluate-n 15 --evaluate-every 10000 --use-early-stopping --compile --mixed-precision bf16 --num-workers 32
```

### brats 2021 

download the dataset
``` bash
./scripts/prepare_brats2021.sh
```

run training on the brats 2021 dataset

``` bash
python train_brats.py --config configs/config_mmDiT_brats21.json --name Run_1 --batch-size 32 --max-epochs 300 --wandb-project AVSS-brats2021 --sample-steps 5 --evaluate-n 15 --evaluate-every 25000 --use-early-stopping --compile --mixed-precision bf16
```



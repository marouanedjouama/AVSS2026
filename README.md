create the conda environment
``` bash
conda env create -f environment.yml
conda activate rf_hourglass
```

install natten for neighborhood attention
``` bash
pip install natten==0.21.5+torch290cu128 -f https://whl.natten.org
```

### brats 2020
download the dataset
``` bash
python ./utils/get_brats2020.py
```

run training on the brats 2021 dataset

``` bash
python train_brats.py --brats 2020 --config configs/config_mmDiT_brats20.json --name Run_1 --batch-size 32 --max-epochs 300 --wandb-project AVSS-brats2020 --sample-steps 1 --evaluate-n 15 --evaluate-every 10000 --use-early-stopping --compile --mixed-precision bf16
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



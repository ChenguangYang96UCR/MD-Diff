# MD-Diff

## Requirements
- [torch](https://pytorch.org/)
- numpy
- pandas

To install requirements (with out neptune):
```bash
pip install -r requirements.txt
```

## Datasets
* The processed datasets for our project is available at [Google Drive](https://drive.google.com/drive/folders/1hIx0GHBejBkIpxEAq61zyhUogPXFf2fo?usp=sharing).

Please put the processed data under [dataset](/data) folder.

## Train / Evaluate USTD
To train and evaluate models, please run the following command:
```bash
./train.sh [model] [dataset] [attribute] [time_length] [pretrain] [config] [batch] [gpu_ids] [seed]
```

| Task         | Dataset         | Command                                                                                     |
|--------------|-----------------|---------------------------------------------------------------------------------------------|
| Pre-train    | PEMS03          | ./train.sh gwavenet PEMS03 NA 12 NA config1 128 2 2030                                      |
| Pre-train    | PEMSBAY         | ./train.sh gwavenet PEMSBAY NA 12 NA config1 128 2 2030                                     |
| Pre-train    | BJAir           | ./train.sh gwavenet BJAir PM25 12 NA config1 128 2 2030                                     |
| Pre-train    | GZAir           | ./train.sh gwavenet GZAir PM25 12 NA config1 128 2 2030                                     |
| Forecasting  | PEMS03 | ./train.sh stdiffusionfore PEMS03 NA 24 gwavenet_NA_20260108T035504 config_PEMS03 64 2 2030        |
| Forecasting  | PEMSBAY | ./train.sh stdiffusionfore PEMSBAY NA 24 **NAME_OF_PRETRAIN** config_PEMSBAY 64 3 2030      |
| Forecasting  | BJAir | ./train.sh stdiffusionfore BJAir PM25 24 **NAME_OF_PRETRAIN** config_BJAir 64 1 2030        |
| Forecasting  | GZAir | ./train.sh stdiffusionfore GZAir PM25 24 **NAME_OF_PRETRAIN** config_GZAir 64 1 2030        |
| Kriging      | PEMS03 | ./train.sh stdiffusion PEMS03 NA 12 **NAME_OF_PRETRAIN** config_PEMS03 64 2 2030            |
| Kriging      | PEMSBAY | ./train.sh stdiffusion PEMSBAY NA 12 **NAME_OF_PRETRAIN** config_PEMSBAY 64 3 2030          |
| Kriging      | BJAir | ./train.sh stdiffusion BJAir PM25 12 **NAME_OF_PRETRAIN** config_BJAir 64 6 2030            |
| Kriging      | GZAir | ./train.sh stdiffusion GZAir PM25 12 **NAME_OF_PRETRAIN** config_GZAir 64 6 2030            |

Each running will train the model 3 times independently with the random seed increasing by 1. 
The framework will save the best model with the highest validation accuracy and evaluate it on the test set automatically after training. 
All the checkpoints and results will be saved at [checkpoints](checkpoints) folder.
For more training, testing, dataset configurations, please refer to [base_options](options/base_options.py), [train_options](options/train_options.py), [test_options](options/test_options.py), and [dataset_options](options/dataset_options.py).

## Reproduce Our Results
We saved the pretrained checkpoints for our USTD at [Google Drive](https://drive.google.com/drive/folders/1OCgxPe3gwWUjeOT5AWaIWqDS60MHUmhP?usp=sharing).
Download the checkpoints files and put them under [checkpoints](checkpoints) folder.
Each checkpoint file contains *run_test.sh* script. Please run the script to reproduce our results by the following command:
```bash
chmod u+x run_test.sh
./run_test.sh
```
The numerical results will be saved at *metrics.sh* and printed out.
The extrapolation results, ground truth, and the uncertainty estimates (if applicable) will be saved at *results.pkl*.


nohup python train.py \
  --model stdiffusionfore \
  --dataset_mode PEMS03 \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 1 \
  --config config_PEMS03 \
  --pretrain gwavenet_NA_20260108T035504 \
  --save_best \
  --t_len 24 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 64 \
> logs/train_PEMS03.out 2>&1 &

gwavenet_NA_20260108T035504
gwavenet_NA_20260109T064635
gwavenet_NA_20260110T090301

nohup python train.py \
  --model gwavenet \
  --dataset_mode PEMS03 \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 2 \
  --config config1 \
  --pretrain NA \
  --save_best \
  --t_len 12 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 128 \
> logs/train_PEMS03.out 2>&1 &


## PEMS03 pretrain

```shell
nohup python train.py \
  --model gwavenet \
  --dataset_mode PEMS03 \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 2 \
  --config config1 \
  --pretrain NA \
  --save_best \
  --t_len 12 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 128 \
> logs/train_gwavenet_PEMS03.out &

nohup python train.py \
  --model stdiffusionfore \
  --dataset_mode PEMS03 \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 2 \
  --config config_PEMS03 \
  --pretrain gwavenet_NA_20260108T035504 \
  --save_best \
  --t_len 24 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 64 \
> logs/train_stdiffusionfore_PEMS03.out  &

```


## BJAir pretrain

```shell
nohup python train.py \
  --model gwavenet \
  --dataset_mode BJAir \
  --pred_attr PM25 \
  --enable_val \
  --gpu_ids 2 \
  --config config1 \
  --pretrain NA \
  --save_best \
  --t_len 12 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 128 \
> logs/train_gwavenet_BJAir.out &

nohup python train.py \
  --model stdiffusionfore \
  --dataset_mode BJAir \
  --pred_attr PM25 \
  --enable_val \
  --gpu_ids 1 \
  --config config_BJAir \
  --pretrain gwavenet_PM25_20260304T100442 \
  --save_best \
  --t_len 24 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 64 \
> logs/train_stdiffusionfore_BJAir.out  &

```


## GZAir pretrain

```shell
nohup python train.py \
  --model gwavenet \
  --dataset_mode GZAir \
  --pred_attr PM25 \
  --enable_val \
  --gpu_ids 3 \
  --config config1 \
  --pretrain NA \
  --save_best \
  --t_len 12 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 128 \
> logs/train_gwavenet_GZAir.out &

nohup python train.py \
  --model stdiffusionfore \
  --dataset_mode GZAir \
  --pred_attr PM25 \
  --enable_val \
  --gpu_ids 2 \
  --config config_GZAir \
  --pretrain gwavenet_PM25_20260124T080921 \
  --save_best \
  --t_len 24 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 64 \
> logs/train_stdiffusionfore_GZAir.out  &

```

## PEMSBAY train

```shell
nohup python train.py \
  --model gwavenet \
  --dataset_mode PEMSBAY \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 1 \
  --config config1 \
  --pretrain NA \
  --save_best \
  --t_len 12 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 128 \
> logs/train_gwavenet_PEMSBAY.out &

nohup python train.py \
  --model stdiffusionfore \
  --dataset_mode PEMSBAY \
  --pred_attr NA \
  --enable_val \
  --gpu_ids 1 \
  --config config_PEMSBAY \
  --pretrain gwavenet_NA_20260124T170537 \
  --save_best \
  --t_len 24 \
  --seed 2030 \
  --eval_epoch_freq 5 \
  --num_train_target 3 \
  --num_threads 4 \
  --batch_size 64 \
> logs/train_stdiffusionfore_PEMSBAY.out  &

```
# Tutorial


## For Training 

0. Change `saved_dir` and `dataset_dir` in `common/config.py`

1. Create new folders under `saved_dir`

e.g. 
```
mkdir log
mkdir finetune
mkdir finetune/model
mkdir pretrain
mkdir pretrain/model
```

2. Move the pretrained model checkpoint and gene2token file to `saved_dir/pretrain/model`

3. Move example datasets and graphs to `dataset_dir`

4. Run training scripts

```sh
bash launch_train.sh
```

## For Testing

```sh
python start_validate.py -c <STEP TO EVALUATE>
```


## For Results Loading

```sh
python load_results.py -c <STEP TO EVALUATE>
```


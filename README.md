# Documentation

Sample command to generate synthetic dataset:

python gen_synth.py --config_id 0  --dataset_num 1 --sample_size 500 --feature_size 20 --causal_num 2 --confounders_num 1 --output_root "./synth_ds"  --p 0.3 

<br />

Sample command to call the train script:

python train.py --epochs 5000 --batch_size 64 --experiment_number 1004  --dataset_config_id 10  --dataset_index 0 --l1_reg 0.01 --l2_reg 0.01 

<br />

Sample command to explain the trained model:

python explain.py  --experiment_number 26 --fold_id 0  --input_size 20 --dataset_config_id 0  --dataset_index 0

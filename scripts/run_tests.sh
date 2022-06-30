#nohup python3 run.py --devices 3 --train_epochs 50 --model CycleGan --model_id noaug --do_predict --wandb > nohup_cg_noaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model ResCycleGan --model_id noaug --do_predict --wandb > nohup_rcg_noaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model CycleGan --augment color translation cutout --model_id allaug --do_predict --wandb > nohup_cg_allaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model ResCycleGan --augment color translation cutout --model_id allaug --do_predict --wandb > nohup_rcg_allaug.out &

python3 run.py --devices 1 --batch_size 16 --train_epochs 7 --model CycleGan --ds_augment --model_id bs16-monetos-dsaug --wandb &&
python3 run.py --devices 1 --batch_size 16 --train_epochs 7 --model CycleGan --diffaugment color translation cutout --model_id bs16-monetos-alldifaug --wandb  &&
python3 run.py --devices 1 --batch_size 16 --train_epochs 8 --model CycleGan  --model_id bs16-monetos-noaug --wandb &&
python3 run.py --devices 1 --batch_size 1 --steps_per_epoch 600 --train_epochs 100 --model CycleGan --cycle_noise 0.2 --model_id bs1-monetos2-noaug-adnoise --wandb ₪₪
python3 run.py --devices 1 --batch_size 16 --train_epochs 8 --model CycleGan --cycle_noise 0.2 --model_id bs16-monetos-noaug-adnoise --wandb



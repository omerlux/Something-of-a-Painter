#nohup python3 run.py --devices 3 --train_epochs 50 --model CycleGan --model_id noaug --do_predict --wandb > nohup_cg_noaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model ResCycleGan --model_id noaug --do_predict --wandb > nohup_rcg_noaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model CycleGan --augment color translation cutout --model_id allaug --do_predict --wandb > nohup_cg_allaug.out &
#
#nohup python3 run.py --devices 3 --train_epochs 50 --model ResCycleGan --augment color translation cutout --model_id allaug --do_predict --wandb > nohup_rcg_allaug.out &


python3 run.py --devices 1 --batch_size 16 --train_epochs 10 --model CycleGan --diffaugment color translation cutout --model_id bs16-monetos-alldifaug --wandb  &&
python3 run.py --devices 1 --batch_size 16 --train_epochs 10 --model CycleGan --ds_augment --model_id bs16-monetos-dsaug --wandb &&
python3 run.py --devices 1 --batch_size 16 --train_epochs 10 --model CycleGan  --model_id bs16-monetos-noaug --wandb


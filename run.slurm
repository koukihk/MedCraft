docker build -t medcraft-medcraft .

docker run -it --rm \
  -v /root/st/MedCraft:/app \
  -v /root/st/MedCraft/dataset:/dataset:ro \
  -v medcraft-venv:/venv \
  --shm-size=16G \
  --ipc=host \
  -v /dev/shm:/dev/shm \
  --gpus '"device=0"' \
  --user $(id -u):$(id -g) \
  --entrypoint /bin/bash \
  medcraft-medcraft

train_path=/dataset
val_path=/dataset/Task03_Liver
fold=3
dist=$((RANDOM % 99999 + 10000))
CUDA_VISIBLE_DEVICES=0 nohup python3.8 -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --ellipsoid --cutmix --logdir="runs/cutmix_bpt_hard_fast.unet$fold" --train_dir $train_path --val_dir $val_path --json_dir datafolds/fold_$fold.json &

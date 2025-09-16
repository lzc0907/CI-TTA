dataset='office-home'
# dataset="PACS"
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx') 
test_envs=3
gpu_ids=1
data_dir='/home/lzc/transferlearning/code/DeepDG_2/data/P_edge/'
max_epoch=120
net='resnet50'
task='img_dg'


i=1
lr=0.001

# erm
# python eval_tta.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output p_2/resnet18/s/erm2 \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[i]} --lr $lr 

python eval_tta_single.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output p_2/resnet18/s/erm2 \
--test_envs 2 --dataset $dataset --algorithm ${algorithm[i]} --lr $lr --gpu_id $gpu_ids

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/c/erm \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[i]} --lr $lr 

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/p/erm \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[i]} --lr $lr 

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/s/erm \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[i]} --lr $lr 

# dann

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/a/dann \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[2]} --lr $lr --alpha 0.5

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/c/dann \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[2]} --lr $lr --alpha 1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/p/dann \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[2]} --lr $lr --alpha 0.1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/s/dann \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[2]} --lr $lr --alpha 0.1

# # mixup
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/a/mixup \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[4]} --lr 0.001 --mixupalpha 0.1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/c/mixup \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[4]} --lr 0.001 --mixupalpha 0.2

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/p/mixup \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[4]} --lr 0.001 --mixupalpha 0.2

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/s/mixup \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[4]} --lr 0.001 --mixupalpha 0.2

# rsc
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/a/rsc \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[3]} --lr $lr --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.3

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/c/rsc \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[3]} --lr $lr --rsc_f_drop_factor 0.3 --rsc_b_drop_factor 0.1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/p/rsc \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[3]} --lr $lr --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/s/rsc \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[3]} --lr $lr --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1

# # vrex
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/a/vrex \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[-1]} --lr $lr --lam 1 --anneal_iters 5000

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/c/vrex \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[-1]} --lr $lr --lam 1 --anneal_iters 100

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/p/vrex \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[-1]} --lr $lr --lam 0.3 --anneal_iters 5000

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_nopre/resnet18/s/vrex \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[-1]} --lr $lr --lam 1 --anneal_iters 10


# MMD
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/a/mmd \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[-3]} --lr $lr --mmd_gamma 10

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/c/mmd \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[-3]} --lr $lr --mmd_gamma 1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/p/mmd \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[-3]} --lr $lr --mmd_gamma 0.5

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/s/mmd \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[-3]} --lr $lr --mmd_gamma 0.5

# #coral
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/a/coral \
# --test_envs 0 --dataset $dataset --algorithm ${algorithm[-2]} --lr $lr --mmd_gamma 0.5

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/c/coral \
# --test_envs 1 --dataset $dataset --algorithm ${algorithm[-2]} --lr $lr --mmd_gamma 0.1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/p/coral \
# --test_envs 2 --dataset $dataset --algorithm ${algorithm[-2]} --lr $lr --mmd_gamma 1

# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output pacs_/resnet18/s/coral \
# --test_envs 3 --dataset $dataset --algorithm ${algorithm[-2]} --lr $lr --mmd_gamma 0.01

# # difex
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18  --task img_dg --output pacs_/resnet18/a/difex \
# --test_envs 0 --dataset PACS --algorithm DIFEX --lr $lr --alpha 0.1 --beta 0.1 --lam 0 --disttype norm-1-norm

# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18  --task img_dg --output pacs_/resnet18/c/difex \
# --test_envs 1 --dataset PACS --algorithm DIFEX --lr $lr --alpha 0.001 --beta 1 --lam 0.01 --disttype norm-1-norm

# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output pacs_/resnet18/p/difex \
# --test_envs 2 --dataset PACS --algorithm DIFEX --lr $lr --alpha 0.001 --beta 0.5 --lam 0.1 --disttype norm-1-norm

# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output pacs_/resnet18/s/difex \
# --test_envs 3 --dataset PACS --algorithm DIFEX --lr $lr --alpha 0.01 --beta 10 --lam 1 --disttype norm-1-norm





# # MLDG 
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
# --test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10

# # Group_DRO
# python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
# --test_envs 0 --dataset PACS --algorithm GroupDRO --groupdro_eta 1 

# # ANDMask
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output /home/lw/lw/test00 \
# --test_envs 0 --dataset PACS --algorithm ANDMask --tau 1 

# # The following experiments are running on the singularity cluster of MSRA.The environment are shown in the following file.
# # CUDA version, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env1.txt
# # GPU information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env2.txt
# # python package information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env.txt
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-0 \
# --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.01 --lam 0.1 --disttype 2-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-1 \
# --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0 --lam 0.01 --disttype 2-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-2 \
# --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.01 --lam 0.01 --disttype 2-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-3 \
# --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 1 --lam 0 --disttype 2-norm

# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-0 \
# --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.1 --lam 0 --disttype norm-1-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-1 \
# --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 1 --lam 0.01 --disttype norm-1-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-2 \
# --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.5 --lam 0.1 --disttype norm-1-norm
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-3 \
# --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 10 --lam 1 --disttype norm-1-norm

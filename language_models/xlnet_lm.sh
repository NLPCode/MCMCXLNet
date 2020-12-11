python -m torch.distributed.launch --nproc_per_node=2 xlnet_lm.py\
    --gpu 0,1 \
    --is_forward 1  \
    --train 1

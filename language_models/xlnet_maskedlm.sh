python -m torch.distributed.launch --nproc_per_node=2  --master_port 29501 xlnet_maskedlm.py\
    --gpu 0,1 \

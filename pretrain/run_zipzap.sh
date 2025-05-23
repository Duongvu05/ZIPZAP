python3 /home/phinn/BERT4ETH/ZIPZAP/pretrain/run_pretrain.py --bizdate="zipzap_exp" --ckpt_dir="zipzap_exp"

python3 run_embed.py --bizdate="zipzap_exp" \
                       --init_checkpoint="zipzap_exp/epoch_50.pth"
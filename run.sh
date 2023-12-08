python main.py \
--model wideresnet \
--lr 0.001 \
--wd 1e-4 \
--lambda-ce 1 \
--lambda-kl 1 \
--train-batch-size 32 \
--gpu 4 \
--seed 2 \
--epochs 100 \
--ablation CLIP-Lung-2class \
--dataset lidc \
--num-labels full \
--ema-alpha 0.999 \
-ema \
-lang \
-save \

























 





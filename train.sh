python train.py --wandb_run_name dqn_state_transform  --use_wandb --batch_size 128 --n_episode 10000 --tau 0.5
git add .
git commit -m "Convert multihot vector to discrete value"
git push
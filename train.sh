python train.py --wandb_run_name dqn_state_transform  --use_wandb --batch_size 128 --n_episode 10000
git add .
git commit -m "Normalize taxi position and convert multihot vector to discrete value"
git push
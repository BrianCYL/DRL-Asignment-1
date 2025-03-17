python train.py --wandb_run_name dqn_eps_0.01  --use_wandb --batch_size 256 --n_episode 10000 --tau 0.5 --eps_end 0.01
git add .
git commit -m "Convert multihot vector to discrete value"
git push
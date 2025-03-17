python train.py --wandb_run_name dqn_norm_grid_size --use_wandb --batch_size 256 --n_episode 15000 --tau 0.3 --eps_end 0.01 --alpha 5e-4
git add .
git commit -m "Normalize grid size"
git push
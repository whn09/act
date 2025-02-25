# conda env create -f conda_env.yaml
# source activate aloha
# pip install opencv-python

# sudo apt-get update
# sudo apt-get install -y \
#     libegl1-mesa \
#     libegl1-mesa-dev \
#     libgl1-mesa-glx \
#     libgles2-mesa-dev

mkdir -p sim ckpt

export MUJOCO_GL=egl

python record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir sim/sim_transfer_cube_scripted \
--num_episodes 50

python visualize_episodes.py \
--dataset_dir sim/sim_transfer_cube_scripted \
--episode_idx 0

python imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir ckpt/sim_transfer_cube_scripted \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0


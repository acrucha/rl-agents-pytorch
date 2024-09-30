#!/bin/sh
#SBATCH --job-name=vss-attacker-ddpg
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 16
#SBATCH -o vss-attacker-ddpg.log
#SBATCH --output=vss-attacker-ddpg-output.txt
#SBATCH --error=vss-attacker-ddpg-error.txt

module load Python/3.10

source $HOME/vss-rl-agents/bin/activate

python $HOME/robocin/vss-rl-agents/train_ddpg.py --env "Pid-v0" --name "PIDTuning-DDPG" --track
python $HOME/robocin/vss-rl-agents/train_ddpg.py --env "Penalty-v0" --name "Penalty-DDPG" --track
python $HOME/robocin/vss-rl-agents/train_ddpg.py --env "Attacker-v0" --name "VSSEF-DDPG" --track
python $HOME/robocin/vss-rl-agents/train_ddpg.py --env "VSS-v0" --name "VSS-DDPG" --track

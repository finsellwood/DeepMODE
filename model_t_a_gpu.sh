#$-q gpu.q@lxbgpu* -l h_rt=24:00:00 
#$-m ea -M fjo18@ic.ac.uk

cd ~/CMSSW_10_2_19/DeepMODE
# export PYTHONPATH=/home/hep/fjo18/.local/lib/python2.7/site-packages:$PYTHONPATH

conda activate icenet
source ~/CMSSW_10_2_19/icenet/setenv.sh
# source /vols/grid/cms/setup.sh
# eval `scramv1 runtime -sh`

python3.9 Python_files/B_Training/train_and_analyse.py

conda deactivate
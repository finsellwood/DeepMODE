#$-q gpu.q@lxcgpu* -l h_rt=3:00:00 

cd ~/CMSSW_10_2_19/DeepMODE/
# export PYTHONPATH=/home/hep/fjo18/.local/lib/python2.7/site-packages:$PYTHONPATH

conda activate icenet
source ~/CMSSW_10_2_19/icenet/setenv.sh
# source /vols/grid/cms/setup.sh
# eval `scramv1 runtime -sh`

python3.9 Python_files/C_Analysis/MultiEventAnnotation.py

conda deactivate

#$-q hep.q -l h_rt=3:00:00 -l h_vmem=24G 

cd ~/CMSSW_10_2_19/DeepMODE/
# export PYTHONPATH=/home/hep/fjo18/.local/lib/python2.7/site-packages:$PYTHONPATH
source ~/CMSSW_10_2_19/icenet/setenv.sh

conda activate icenet
# source /vols/grid/cms/setup.sh
# eval `scramv1 runtime -sh`

python3.9 Python_files/D_Other_Files/df_tf_test.py

conda deactivate
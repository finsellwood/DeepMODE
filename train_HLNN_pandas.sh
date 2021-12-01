#$-q gpu.q@lxcgpu* -l h_rt=24:00:00
#$-m ea -M fjo18@ic.ac.uk

cd ~/CMSSW_10_2_19/src/UserCode/DeepLearning
export PYTHONPATH=/home/hep/fjo18/.local/lib/python3.6/site-packages:$PYTHONPATH
conda activate icenet
source ~/CMSSW_10_2_19/icenet/setenv.sh
source /vols/grid/cms/setup.sh

eval `scramv1 runtime -sh`

python3 Python_files/train_HLNN_pandas.py

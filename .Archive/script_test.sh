#$-q gpu.q@lxbgpu* -l h_rt=24:00:00

cd ~/CMSSW_10_2_19/src/UserCode/DeepLearning
export PYTHONPATH=/home/hep/fjo18/.local/lib/python3.6/site-packages:$PYTHONPATH

source /vols/grid/cms/setup.sh
eval `scramv1 runtime -sh`

pip3 install --user pip --upgrade
pip3 install --user awkward --upgrade
pip3 install --user vector

python3 testcode.py
 

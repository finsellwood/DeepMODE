#$-q gpu.q@lxcgpu* -l h_rt=24:00:00

cd ~/CMSSW_10_2_19/src/UserCode/DeepLearning
export PYTHONPATH=/home/hep/fjo18/.local/lib/python3.6/site-packages:$PYTHONPATH

source /vols/grid/cms/setup.sh
eval `scramv1 runtime -sh`

pip3 install --user pip --upgrade
pip3 install --user awkward --upgrade
pip3 install --user vector
pip3 install --user uproot3
pip3 install --user tensorflow --upgrade
pip3 install --user pandas --upgrade
pip3 install --user root_numpy --upgrade
## a script for setting up the missing commans - to be run once 

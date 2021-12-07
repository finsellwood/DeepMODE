#$-q hep.q -l h_rt=3:00:00 -l h_vmem=24G 
#$-m ea -M fjo18@ic.ac.uk

cd ~/CMSSW_10_2_19/src/UserCode/DeepLearning
export PYTHONPATH=/home/hep/fjo18/.local/lib/python3.6/site-packages:$PYTHONPATH

source /vols/grid/cms/setup.sh
source /vols/software/cuda/setup.sh 10.2.2
eval `scramv1 runtime -sh`

python3 Python_files/create_dataset_pd.py
 

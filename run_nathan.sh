#$-q hep.q -l h_rt=24:00:00 
#$-m ea -M fjo18@ic.ac.uk

cd ~/CMSSW_10_2_19/src/UserCode/DeepLearning

conda activate icenet
source ~/CMSSW_10_2_19/icenet/setenv.sh

python3.9 Python_files/B_Training/nathan_main_file.py
 
conda deactivate
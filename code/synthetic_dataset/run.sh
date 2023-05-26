# data processing
mkdir ../../data/synthetic_dataset
cd ../../data/synthetic_dataset
mkdir train_data
mkdir test_data

python synthesis_dataset_generation.py

# model training
cd ../../code/synthetic_dataset
python -u FDN.py
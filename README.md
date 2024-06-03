# Building detection

## Environment setup

First create a conda environment and install the required libraries:
```bash
conda create --name building_detection python=3.12.2

pip3 install -r requirements.txt

# install torch with cuda support. Change the cuda version accordingly to the local setup

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Used dataset
SpaceNet 7: https://spacenet.ai/sn7-challenge/

The data is stored in aws, as long as the user has aws cli installed, it can be downloaded with the following code

Training data 
```bash
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz . 

aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz . 
```

Testing data
```bash
aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz . 
```


# Steps to generate the dataset in our format
1. Download the data mentioned above.
1. Split data into train, valid and test using the following notebook: `./notebooks/split_dataset_v2.ipynb`
1. Convert the ground truth masks to png with the following notebook: `./notebooks/generate_masks.ipynb`
1. (Optional) Create dataframe for test set with the following notebook: `./notebooks/create_public_test_df.ipynb`


# Training code example
In order to be able to train, one should create the necessary environment, download and convert the data set to our format, both as explained above. Another requirement, prior to training, is to set up the correct paths in the config.ini file.
```bash
python train.py
```

# Additional files
The used dataset split and our best experiments model & results can be found here: https://drive.google.com/file/d/142aCFIZ7EMuatwBxDcyC9Ec2kfhMyK7c/view?usp=sharing
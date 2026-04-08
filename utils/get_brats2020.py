import os
# # Create the directory if it doesn't exist
# os.makedirs('./data', exist_ok=True)

print("downloading the dataset...")
# # Download the BraTS 2020 dataset to ./data/
# os.system('curl -L -o ./data/brats20.zip https://www.kaggle.com/api/v1/datasets/download/awsaf49/brats20-dataset-training-validation')

print("unziping the dataset...")
# # Unzip the dataset
# os.system('unzip -q ./data/brats20.zip -d ./data/brats20')

print("removing corrupted cases...")
# remove case 355
os.system('rm -rf ./data/brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355')

print("dataset is ready!")
import splitfolders
import os

# Define dataset directory
dataset_path = r"C:\Users\Balir\Downloads\Eye Disease Detection\training\dataset"  # Update this path if your dataset is elsewhere
output_path = "output"  # Output directory for train/test split

# Create train and test folders
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Split the dataset (80% train, 20% validation)
splitfolders.ratio(dataset_path, output=output_path, seed=1337, ratio=(0.8, 0.2))

print("Dataset successfully split into training and validation sets!")

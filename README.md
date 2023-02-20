# NFT-Recommender-RecBole

### config
- The user-defined parameters needed for the experimental setup are stored.
  - e.g., data path, epoch, batch size, metrics, 
- File name: {model}_{collection}_{item_feature}.config

### dataset
- **collections (MAIN)**
  - For each collection, there are .inter and .itememb files included.
  - The .inter file represents user-item interactions, and the .itememb file represents item embeddings.
- transactions
  - For each collection, user-item interactions are included.
  - It is used for creating the .inter file.
- item_features
  - For each collection, image, text, and price features are included.
  - It is used to create the .itememb file.

### hyper
- Used for hyperparameter optimization.
- hyperparameter search ranges are included.
- File name: {model}.hyper

### hyper_result
- Contains the results of hyperparameter optimization.
- The .best_params file contains the optimal hyperparameters that are saved after the hyperparameter optimization process.
- The .result file contains the performance of all hyperparameter combinations.

### result
- The results of the performance evaluation on the test set are saved in the result folder.
- File name: {model}.csv

### runfile
- Contains shell scripts that can be used to run the main file.

### saved
- The best model that shows the lowest valid metric during the model training process is saved.

### Create_dataset.ipynb
- Code to create the input data file in the format required by Recbole.

### Create_config.ipynb
- Code to create the configurations in the format required by Recbole.

### main.py (MAIN)
- The code to run experiments using the built-in models in RecBole.

# Covid19 Project

Predicting mutations of covid19 spike protein sequences

## Installation

```
# Run this in the project root
$ conda activate env
(env) $ conda install -r requirements.txt
```

## Usage

Scripts in the folder `./src/scripts/` are where you can run python code train_predict.py directly.


## Project Structure

    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── figure         <- Intermediate data that has been transformed.
    │   ├── model_save     <- Save the training model.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        │
        │
        ├── utils          <- Scripts for grouping scripts below into handy chunks
        │   └── utils.py
        │
		├── scripts       <- training and predicting the dataset  **	
		│	   └── train_predict.py
		│
        ├── data           <- Scripts to read in or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for processing
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models                
        │   └── train_model.py
        │   
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
		
--------


import os 
import pandas as pd
import numpy as np

def create_ll_partition(dataframe_path: str, validation_split=1, ll_split=3,
                        partition_prop=0.2):
    if not os.path.exists(dataframe_path):
        raise ValueError(
            f'{dataframe_path} does not exist yet. Please generate the dataset first.')

    metadata_df = pd.read_csv(dataframe_path)

    # Identify the rows where metadata_df["split"] is 0
    mask = metadata_df["split"] == validation_split

    # Calculate partition_prop of these rows
    n = int(partition_prop * mask.sum())

    # Randomly select n indices from the rows where metadata_df["split"] is train_split
    random_indices = np.random.choice(
        metadata_df[mask].index, size=n, replace=False)

    # Change the value of 'split' column in these rows to ll_split
    metadata_df.loc[random_indices, "split"] = ll_split

    # Save the dataframe to a CSV file
    metadata_df.to_csv(dataframe_path, index=False)

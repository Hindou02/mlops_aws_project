# in this file we will make the preprocessing 

import pandas as pd
import sys
import yaml
import os
import s3fs

## load params from yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))['preprocess']
aws_params = yaml.safe_load(open(params_path))['aws']

# def save_df(df,path,aws_params):
#     # Read CSV from S3
#     storage_options={"key": aws_params['aws_access_key_id'], "secret": aws_params['aws_secret_access_key']}
#     df.to_csv(path,index=False, storage_options=storage_options)
   

# def preprocess(input_path,output_path,aws_params):
#     # os.makedirs(os.path.dirname(output_path),exist_ok=True) # if does not exist

#     data = pd.read_csv(input_path)
#     #? any data preprocessing needed should be done here
#     data = data.dropna()
#     save_df(data,output_path,aws_params)
#     #data.to_csv(output_path,index=False)
#     print(f"Preprocessed data saved to :{output_path}")

# if __name__== "__main__" :
#     preprocess(params["input"],params["output"],aws_params)



# Updated preprocess.py
def save_df(df, path):
    # Create the directory if it doesn't exist locally
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save locally
    df.to_csv(path, index=False)
    print(f"Preprocessed data saved locally to: {path}")

def preprocess(input_path, output_path):
    # If input is S3, pandas can still read it if you have s3fs installed,
    # but the output must be local for DVC to see it.
    data = pd.read_csv(input_path) 
    data = data.dropna()
    save_df(data, output_path)

if __name__ == "__main__" :
    # Remove aws_params if they are no longer needed for local saving
    preprocess(params["input"], params["output"])
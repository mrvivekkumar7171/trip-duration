# make_dataset.py
from sklearn.model_selection import train_test_split
import pathlib, yaml, sys
import pandas as pd

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    data_path = home_dir / sys.argv[1]
    output_path = home_dir / sys.argv[2]
    data_path_str = data_path.as_posix()
    output_path_str = output_path.as_posix()
    pathlib.Path(output_path_str).mkdir(parents=True, exist_ok=True)

    data = load_data(data_path_str)
    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path_str)

if __name__ == "__main__":
    main()
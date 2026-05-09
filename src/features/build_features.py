from feature_definitions import feature_build
import pandas as pd
import pathlib, sys


def load_data(data_path):
    """Load your dataset from a given path"""

    df = pd.read_csv(data_path)
    return df

def save_data(train, test, output_path):
    """Save the split datasets to the specified output path"""

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_path = home_dir / sys.argv[1]

    train_path = input_path / 'train.csv'
    test_path = input_path / 'test.csv'
    train_path_str = train_path.as_posix()
    test_path_str = test_path.as_posix()

    train_data = pd.read_csv(train_path_str)
    test_data = pd.read_csv(test_path_str)

    output_path = home_dir / sys.argv[2]
    output_path_str = output_path.as_posix()

    train_data = feature_build(train_data, 'train')
    test_data = feature_build(test_data, 'test')

    save_data(train_data, test_data, output_path_str)
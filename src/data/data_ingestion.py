# pip install kaggle : To install kaggle library
# kaggle auth login : To authenticate with Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import pathlib, yaml, sys, click, random, os, zipfile
import pandas as pd
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger


logger.info('Fetching Dataset from Kaggle and Splitting into train, test and validation datasets')

class TrainTestCreation:
    """
        Download Kaggle Competition data, unzip, delete zip files, load the train dataset and split into train,
        test and validation datasets. Then save the splitted datasets into the specified output folder.
    """
    def __init__(self, params) -> None:
        """Initialize class variables with provided parameters
        """
        self.seed = params['seed']
        self.test_split = params['test_split']
        self.api = KaggleApi()
        self.api.authenticate()

        logger.info('Call to make_ingestion')
        
    def download_data(self, data_path : str, output_path : str) -> None:
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            competition_name = data_path
            extract_path = output_path
            self.api.competition_download_files(competition_name, path=extract_path)
        except Exception as e:
            logger.error('Downloading dataset failed')
            raise e
        else:
            logger.info('Downloading dataset performed successfully')

    def unzip_data(self, competition_name : str, data_path : str) -> None:
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            with zipfile.ZipFile(os.path.join(data_path, f'{competition_name}.zip'), 'r') as zip_ref:
                zip_ref.extractall(data_path)

            with zipfile.ZipFile(os.path.join(data_path, 'train.zip'), 'r') as zip_ref:
                zip_ref.extractall(data_path)
                
            with zipfile.ZipFile(os.path.join(data_path, 'test.zip'), 'r') as zip_ref:
                zip_ref.extractall(data_path)
        except Exception as e:
            logger.error('File extraction failed')
            raise e
        else:
            logger.info('File extraction performed successfully')

    def clean_zip_data(self, data_path : str) -> None:
        '''This function reads data from input path and stores it into a dataframe'''
        try:

            for file in os.listdir(data_path):
                if file.endswith('.zip'):
                    os.remove(os.path.join(data_path, file))

        except Exception as e:
            logger.error('Removing zip files failed')
            raise e
        else:
            logger.info('Zip file removed successfully')

    def load_data(self, data_path : str) -> None:
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            self.df = pd.read_csv(os.path.join(data_path, 'train.csv'))
        except Exception as e:
            logger.error('Loading data failed')
            raise e
        else:
            logger.info('Data loaded successfully')

    def split_data(self) -> None:
        '''This function splits the whole data into train test as per the test percent provided'''
        try:
            self.train, self.test = train_test_split(self.df, test_size=self.test_split, random_state=self.seed)

            idx_list = list(self.test.index)
            test_idx = random.sample(idx_list, len(idx_list)//2)
            val_idx = list(set(idx_list)- set(test_idx))

            self.validate = self.test.loc[val_idx]
            self.test = self.test.loc[test_idx]
        except Exception as e:
            logger.error('Data split failed')
            raise e
        else:
            logger.info('Data splitted successfully')

    def save_data(self, output_path : str) -> None:
        '''This function writes the data into specified destination folder'''
        try:
            pd.read_csv(output_path + '/test.csv').to_csv(output_path + '/test_kaggle.csv', index=False)
            self.train.to_csv(output_path + '/train.csv', index=False)
            self.test.to_csv(output_path + '/test.csv', index=False)
            self.validate.to_csv(output_path + '/validate.csv', index=False)
        except Exception as e:
            logger.error('Dataset saving failed')
            raise e
        else:
            logger.info('Dataset saved successfully')

    def fit(self, data_path_str : str, output_path_str : str) -> None:

        self.download_data(data_path_str, output_path_str)
        self.unzip_data(data_path_str, output_path_str)
        self.clean_zip_data(output_path_str)
        self.load_data(output_path_str)
        self.split_data()
        self.save_data(output_path_str)

# @click.command() # to turn a function into a CLI tool
# @click.argument('input_filepath', type=click.Path())
# @click.argument('output_filepath', type=click.Path())
def main() -> None:
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["data_ingestion"]

    data_path = sys.argv[1]
    output_path = home_dir / sys.argv[2]
    output_path_str = output_path.as_posix()
    pathlib.Path(output_path_str).mkdir(parents=True, exist_ok=True)

    split_data = TrainTestCreation(params)
    split_data.fit(data_path, output_path_str)

if __name__ == '__main__':
    main()
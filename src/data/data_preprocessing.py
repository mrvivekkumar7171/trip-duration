"""
    take data from data/raw and perform preprocessing and save the preprocessed data in data/processed like
    train_processed.csv and test_processed.csv
"""
import pathlib, yaml, sys, click
import pandas as pd
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src.logger import logger

logger.info('Load train, test and validate dataset, perform preprocessing and save')

class TrainTestCreation:

    def __init__(self, params : dict) -> None:
        """Initialize class variables with provided parameters
        """
        self.trip_duration_lowlimit = params['trip_duration_lowlimit']
        self.trip_duration_uplimit = params['trip_duration_uplimit']
        self.pickup_latitude_lowlimit = params['pickup_latitude_lowlimit']
        self.pickup_latitude_uplimit = params['pickup_latitude_uplimit']
        self.dropoff_latitude_lowlimit = params['dropoff_latitude_lowlimit']
        self.dropoff_latitude_uplimit = params['dropoff_latitude_uplimit']
        self.pickup_longitude_lowlimit = params['pickup_longitude_lowlimit']
        self.pickup_longitude_uplimit = params['pickup_longitude_uplimit']
        self.dropoff_longitude_lowlimit = params['dropoff_longitude_lowlimit']
        self.dropoff_longitude_uplimit = params['dropoff_longitude_uplimit']

        logger.info('Call to data_preprocessing')
        
    def load_data(self, data_path : str) -> None:
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            self.train = pd.read_csv(f"{data_path}/train.csv")
            self.test = pd.read_csv(f"{data_path}/test.csv")
            self.validate = pd.read_csv(f"{data_path}/validate.csv")
        except Exception as e:
            logger.error('Loading data failed')
            raise e
        else:
            logger.info('Data loaded successfully')

    def date_type_conversion(self) -> None:
        '''Convert all the date columns from object datatype to datetime datatype'''
        try:
            self.train['pickup_datetime'] = pd.to_datetime(self.train['pickup_datetime'])
            self.train['dropoff_datetime'] = pd.to_datetime(self.train['dropoff_datetime'])
            
            self.test['pickup_datetime'] = pd.to_datetime(self.test['pickup_datetime'])
            self.test['dropoff_datetime'] = pd.to_datetime(self.test['dropoff_datetime'])
            
            self.validate['pickup_datetime'] = pd.to_datetime(self.validate['pickup_datetime'])
            self.validate['dropoff_datetime'] = pd.to_datetime(self.validate['dropoff_datetime'])
        except Exception as e:
            logger.error('Date conversion of columns has failed')
            raise e
        else:
            logger.info('Date conversion performed successfully')

    def outlier_removal(self) -> None:
        '''This function removes the outlier from the data based on upper and lower threshold provided'''
        try:

            self.train = self.train[(self.train['trip_duration'] >= self.trip_duration_lowlimit) & (self.train['trip_duration'] <= self.trip_duration_uplimit)]
            self.train = self.train.loc[(self.train['pickup_latitude'] >= self.pickup_latitude_lowlimit) & (self.train['pickup_latitude'] <= self.pickup_latitude_uplimit)]
            self.train = self.train.loc[(self.train['pickup_longitude'] >= self.pickup_longitude_lowlimit) & (self.train['pickup_longitude'] <= self.pickup_longitude_uplimit)]
            self.train = self.train.loc[(self.train['dropoff_latitude'] >= self.dropoff_latitude_lowlimit) & (self.train['dropoff_latitude'] <= self.dropoff_latitude_uplimit)]
            self.train = self.train.loc[(self.train['dropoff_longitude'] >= self.dropoff_longitude_lowlimit) & (self.train['dropoff_longitude'] <= self.dropoff_longitude_uplimit)]
            
            self.test = self.test[(self.test['trip_duration'] >= self.trip_duration_lowlimit) & (self.test['trip_duration'] <= self.trip_duration_uplimit)]
            self.test = self.test.loc[(self.test['pickup_latitude'] >= self.pickup_latitude_lowlimit) & (self.test['pickup_latitude'] <= self.pickup_latitude_uplimit)]
            self.test = self.test.loc[(self.test['pickup_longitude'] >= self.pickup_longitude_lowlimit) & (self.test['pickup_longitude'] <= self.pickup_longitude_uplimit)]
            self.test = self.test.loc[(self.test['dropoff_latitude'] >= self.dropoff_latitude_lowlimit) & (self.test['dropoff_latitude'] <= self.dropoff_latitude_uplimit)]
            self.test = self.test.loc[(self.test['dropoff_longitude'] >= self.dropoff_longitude_lowlimit) & (self.test['dropoff_longitude'] <= self.dropoff_longitude_uplimit)]
            
            self.validate = self.validate[(self.validate['trip_duration'] >= self.trip_duration_lowlimit) & (self.validate['trip_duration'] <= self.trip_duration_uplimit)]
            self.validate = self.validate.loc[(self.validate['pickup_latitude'] >= self.pickup_latitude_lowlimit) & (self.validate['pickup_latitude'] <= self.pickup_latitude_uplimit)]
            self.validate = self.validate.loc[(self.validate['pickup_longitude'] >= self.pickup_longitude_lowlimit) & (self.validate['pickup_longitude'] <= self.pickup_longitude_uplimit)]
            self.validate = self.validate.loc[(self.validate['dropoff_latitude'] >= self.dropoff_latitude_lowlimit) & (self.validate['dropoff_latitude'] <= self.dropoff_latitude_uplimit)]
            self.validate = self.validate.loc[(self.validate['dropoff_longitude'] >= self.dropoff_longitude_lowlimit) & (self.validate['dropoff_longitude'] <= self.dropoff_longitude_uplimit)]

        except Exception as e:
            logger.error('Outlier removal for data has failed')
            raise e
        else:
            logger.info('Outlier removal performed successfully')

    def save_data(self, output_path : str) -> None:
        '''This function writes the data into specified destination folder'''
        try:
            self.train.to_csv(output_path + '/train_interim.csv', index=False)
            self.test.to_csv(output_path + '/test_interim.csv', index=False)
            self.validate.to_csv(output_path + '/validate_interim.csv', index=False)
        except Exception as e:
            logger.error('Writing data failed')
            raise e
        else:
            logger.info('Write performed successfully')

    def fit(self, data_path_str : str, output_path_str : str) -> None:
        self.load_data(data_path_str)
        self.date_type_conversion()
        self.outlier_removal()
        self.save_data(output_path_str)

# @click.command() # to turn a function into a CLI tool
# @click.argument('input_filepath', type=click.Path())
# @click.argument('output_filepath', type=click.Path())
def main() -> None:
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["data_preprocessing"]

    data_path = home_dir / sys.argv[1]
    output_path = home_dir / sys.argv[2]
    data_path_str = data_path.as_posix()
    output_path_str = output_path.as_posix()
    pathlib.Path(output_path_str).mkdir(parents=True, exist_ok=True)

    split_data = TrainTestCreation(params)
    split_data.fit(data_path_str, output_path_str)

if __name__ == '__main__':
    main()
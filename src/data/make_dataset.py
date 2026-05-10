# make_dataset.py
from sklearn.model_selection import train_test_split
import pathlib, yaml, sys, click, random
from src.logger import infologger
import pandas as pd

infologger.info('Basic cleaning and Splitting into train test data from the whole data')

class TrainTestCreation:

    def __init__(self, params):
        """Initialize class variables with provided parameters
        """
        self.seed = params['seed']
        self.test_split = params['test_split']
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

        infologger.info(f'Call to make_dataset with the parameters: Test Percentage: {self.test_split}, and seed value: {self.seed}')
        
    def load_data(self, data_path):
        '''This function reads data from input path and stores it into a dataframe'''
        try:
            self.df = pd.read_csv(data_path)
        except Exception as e:
            infologger.info(f'Reading failed with error: {e}')
        else:
            infologger.info('Read performed successfully')

    def date_type_conversion(self):
        '''Convert all the date columns from object datatype to datetime datatype'''
        try:
            self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'])
            self.df['dropoff_datetime'] = pd.to_datetime(self.df['dropoff_datetime'])
        except Exception as e:
            infologger.info(f'Date conversion of columns has failed with error : {e}')
        else:
            infologger.info('Date conversion performed successfully')

    def outlier_removal(self):
        '''This function removes the outlier from the data based on upper and lower threshold provided'''
        try:
            self.df = self.df[(self.df['trip_duration'] >= self.trip_duration_lowlimit) & (self.df['trip_duration'] <= self.trip_duration_uplimit)]
            self.df = self.df.loc[(self.df['pickup_latitude'] >= self.pickup_latitude_lowlimit) & (self.df['pickup_latitude'] <= self.pickup_latitude_uplimit)]
            self.df = self.df.loc[(self.df['pickup_longitude'] >= self.pickup_longitude_lowlimit) & (self.df['pickup_longitude'] <= self.pickup_longitude_uplimit)]
            self.df = self.df.loc[(self.df['dropoff_latitude'] >= self.dropoff_latitude_lowlimit) & (self.df['dropoff_latitude'] <= self.dropoff_latitude_uplimit)]
            self.df = self.df.loc[(self.df['dropoff_longitude'] >= self.dropoff_longitude_lowlimit) & (self.df['dropoff_longitude'] <= self.dropoff_longitude_uplimit)]
        except Exception as e:
            infologger.info(f'Outlier removal for data has failed with error : {e}')
        else:
            infologger.info(f'Outlier removal performed successfully')

    def split_data(self):
        '''This function splits the whole data into train test as per the test percent provided'''
        try:
            self.train, self.test = train_test_split(self.df, test_size=self.test_split, random_state=self.seed)

            idx_list = list(self.test.index)
            test_idx = random.sample(idx_list, len(idx_list)//2)
            val_idx = list(set(idx_list)- set(test_idx))

            self.validate = self.test.loc[val_idx]
            self.test = self.test.loc[test_idx]
        except Exception as e:
            infologger.info(f'Splitting failed with error: {e}')
        else:
            infologger.info('Split performed successfully')

    def save_data(self, output_path):
        '''This function writes the data into specified destination folder'''
        try:
            self.train.to_csv(output_path + '/train.csv', index=False)
            self.test.to_csv(output_path + '/test.csv', index=False)
            self.validate.to_csv(output_path + '/validate.csv', index=False)
        except Exception as e:
            infologger.info(f'Writing data failed with error: {e}')
        else:
            infologger.info('Write performed successfully')

    def fit(self, data_path_str, output_path_str):

        self.load_data(data_path_str)
        self.date_type_conversion()
        self.outlier_removal()
        self.split_data()
        self.save_data(output_path_str)

# @click.command() # to turn a function into a CLI tool
# @click.argument('input_filepath', type=click.Path())
# @click.argument('output_filepath', type=click.Path())
def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    data_path = home_dir / sys.argv[1]
    output_path = home_dir / sys.argv[2]
    data_path_str = data_path.as_posix()
    output_path_str = output_path.as_posix()
    pathlib.Path(output_path_str).mkdir(parents=True, exist_ok=True)

    split_data = TrainTestCreation(params)
    split_data.fit(data_path_str, output_path_str)

if __name__ == '__main__':
    main()
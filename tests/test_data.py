from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features.feature_definitions import feature_build
from src.features.build_features import BuildFeatures
from src.logger import logger
import unittest, os, pickle, yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TestDemo(unittest.TestCase):
    """
        It fetchs a pre-trained ML model and validation dataset from cloud storage (AWS S3), process the
        test data, generate predictions, calculate how accurate the model is, and finally output a bar chart
        visualizing the performance metrics.
    """

    def __init__(self):
        """connect to an AWS S3 bucket and download essential files. If the downloads fail, it logs an
        error. If they succeed, it immediately loads the CSV data into a Pandas DataFrame for processing.
        """
        try:
            s3 = boto3.client("s3")
            s3.download_file(Bucket="nyctrip-bucket", Key="validate_data.csv", Filename="validate_data.csv")
            s3.download_file(Bucket="nyctrip-bucket", Key="test_bestmodel.pkl", Filename="test_bestmodel.pkl")
            s3.download_file(Bucket="nyctrip-bucket", Key="loc_kmeans.pkl", Filename="loc_kmeans.pkl")
        except Exception as e:
            logger.error('Not able to Connect to S3')
            raise e
        else:
            self.df = pd.read_csv("validate_data.csv")

    def predict(self):
        """Prepares the data and generates predictions
        """
        # building a features for prediction on validate data 
        # seperating input and output data
        # initializing a object of predictor class
        # generating predictions
        feat = BuildFeatures()
        self.df = feat.build(self.df, 'loc_kmeans.pkl') 
        features = yaml.safe_load(open('params.yaml'))['train_model']['features']
        self.df = self.df[features]

        target = 'trip_duration'
        self.x = self.df.drop(columns=[target])
        self.y = self.df[target]

        predictor = pickle.load(open('test_bestmodel.pkl', 'rb'))

        self.y_pred = predictor.predict(self.x)

    def score(self):
        rmse = round(np.sqrt(mean_squared_error(self.y, self.y_pred)),2)
        mae = round(mean_absolute_error(self.y, self.y_pred),2)
        # root mean square percentage error
        rmspe = round(np.sqrt(np.sum(np.power(((self.y-self.y_pred)/self.y), 2))/len(self.y))*100, 3)
        r2 = round(r2_score(self.y, self.y_pred)*100, 2)
        # adjusted_r2_score
        adjr2 = round(1-(1-r2_score(self.y, self.y_pred))*((self.x.shape[0]-1)/(self.x.shape[0]-self.x.shape[1]-1)),2)
        #dictionarself.y storing all these testing score and this will be the returning value of function
        self.score_dict = {
            'Root Mean Square Error':rmse,
            'Mean Absolute Error':mae,
            'Root Mean Square Percentage Error':rmspe,
            'R2 Score':r2,
            'Adjusted R2 Score':adjr2
            }
        return self.score_dict

    def test(self):
        try:
            self.predict()
            self.score()
            fig, ax = plt.subplots()
            ax.bar(list(self.score_dict.keys()), list(self.score_dict.values()))
            ax.set_ylabel('Score')
            ax.set_xlabel('Metrices')
            ax.set_title('Different Scoring Metrices for model')
            plt.xticks(rotation = 'vertical')
            plt.savefig('metrices_bars.png')
            plt.close(fig) # Good practice to close figures in tests
            logger.info(f"Image saved successfully at: {os.path.abspath('metrices_bars.png')}")
        except Exception as e:
            logger.error('Error in plotting and predicting')
            raise e
        else:
            logger.info('Plotted successful')

if __name__ == '__main__':
    unittest.main()
import os
import sys

from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from data_transformation import DataTransformationConfig
from data_transformation import DataTransformation

@dataclass
class DataIngestionCongig:
    train_data_path=os.path.join("artifacts",'train.csv')
    test_data_path=os.path.join("artifacts",'test.csv')
    raw_data_path=os.path.join("artifacts",'raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionCongig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')

        try:
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initaite_data_transformation(train_data,test_data)

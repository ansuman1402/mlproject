import os
import sys

from contourpy.util import data
sys.path.append("d:\Projects\mlproject\src")
from components import data_transformation
from exceptions import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig
from components.model_trainer import ModelTrainerConfig, ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact',"train.csv")
    test_data_path: str = os.path.join('artifact',"test.csv")
    raw_data_path: str = os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            df= pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as DataFrame.")
            #print (df.head())

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)

            logging.info("Train_Test_split initiated.")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)

            logging.info("Ingestion of the data is completed.")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path) 
        except Exception as e:
            raise CustomException(e,sys) # type: ignore
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.intiate_data_ingestion()

    data_transform = DataTransformation()
    train_arr,test_arr,_=data_transform.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
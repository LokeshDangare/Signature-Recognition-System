import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts, ModelPusherArtifacts

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
        try:
            logging.info("Getting the dataset from GCloud Storage Bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the dataset from GCloud Storage")
            logging.info("Exited the start_data_ingestion method of TrainingPipeline class")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_data_transformation method of TrainingPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifacts = (data_transformation.initiate_data_transformation())
            logging.info("Exited the start_data_transformation method of TrainingPipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("Entered the start_model_trainer method of TrainingPipeline class")
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config = self.model_trainer_config,
            )
            model_trainer_artifacts = (model_trainer.initiate_model_trainer())
            logging.info("Exited the start_model_trainer method of TrainingPipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts,
                               data_transformation_artifacts: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Entered the start_model_evaluation method of TrainingPipeline class")
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_artifacts=model_trainer_artifacts
            )
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of TrainingPipeline class")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_pusher(self, model_trainer_artifacts: ModelTrainerArtifacts) -> ModelPusherArtifacts:
        logging.info("Entered the start_model_pusher method of TrainingPipeline class")
        try:
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config, model_trainer_artifacts=model_trainer_artifacts)
            model_pusher_artifacts = model_pusher.initiate_model_pusher()
            logging.info("Initiated the model pusher")
            logging.info("Exited the start_model_pusher method of TrainingPipeline class")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainingPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts=data_ingestion_artifacts)
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifacts=data_transformation_artifacts)
            model_evaluation_artifacts =self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts, data_transformation_artifacts=data_transformation_artifacts)
            if not model_evaluation_artifacts.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifacts = self.start_model_pusher(model_trainer_artifacts=model_trainer_artifacts)

        except Exception as e:
            raise CustomException(e, sys) from e

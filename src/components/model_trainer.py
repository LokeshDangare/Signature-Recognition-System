import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from src.logger import logging
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_trainer_config: Configuration for model trainer
        :param data_transformation_artifacts: Artifacts for data transformation
        """
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS

    def train(self, model, criterion, optimizer, train_dataloader, valid_dataloader):
        """
        Method Name: train
        Description: This method takes pretrained model, loss, optimizer, train and valid data laoder
        to start training
        """
        try:
            total_train_loss = 0
            total_test_loss = 0

            model.train()
            with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description(f'training')
                for images, idxs in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    idxs = idxs.to(DEVICE, non_blocking=True)
                    output = model(images)

                    loss = criterion(output, idxs)
                    total_train_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()
            with tqdm(valid_dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description(f'testing')
                for images, idxs in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    idxs = idxs.to(DEVICE, non_blocking=True)
                    output = model(images)

                    loss = criterion(output, idxs)
                    total_test_loss += loss.item()

            train_loss = total_train_loss / len(self.data_transformation_artifacts.train_transformed_object)
            valid_loss = total_test_loss / len(self.data_transformation_artifacts.valid_transformed_object)
            print(f'Train Loss: {train_loss:.4f} Test Loss: {valid_loss:.4f}')

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        """
        Method Name: initiate_model_trainer
        Description: This method initiate model trainer steps

        Output: Return model trainer artifacts
        On Failur: Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the initiate_model_trainer method of Model trainer class")

            train_dataset = load_object(self.data_transformation_artifacts.train_transformed_object)
            valid_dataset = load_object(self.data_transformation_artifacts.valid_transformed_object)

            logging.info("Loaded dataset from data transformation artifacts")
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
            valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
            logging.info("Loaded train and valid data loader")

            model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
            logging.info("Loaded pretrained resnet34 model")

            model.fc = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(model.fc.in_features, self.data_transformation_artifacts.classes)
            )
            logging.info("Updated the last layer of pretrained model")

            model = model.to(DEVICE)

            criterion = torch.nn.CrossEntropyLoss()
            logging.info("Cross entropy loss function is used.")

            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
            logging.info("SGD optimizer is used.")

            logging.info("Model Training Started")
            for i in range(self.epochs):
                logging.info(f"Model training at epoch: {i+1}")
                print(f"Epoch: {i+1}/{self.epochs}")
                self.train(model, criterion, optimizer, train_loader, valid_loader)
            logging.info("Model Training Done!!!")

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Saved trained model at {self.model_trainer_config.TRAINED_MODEL_PATH}")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Model trainer artifacts: {model_trainer_artifacts}")

            logging.info("Exited the initiate_model_trainer method of Model trainer class")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e


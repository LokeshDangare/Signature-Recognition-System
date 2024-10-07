import os
import sys
import torch
from PIL import Image
from src.constants import *
from src.logger import logging
from torchvision import transforms
from src.exception import CustomException
from src.utils.main_utils import read_yaml_file
from src.configurations.gcloud_syncer import GCloudSync


class PredictionPipeline:

    def __init__(self):
        self.gcloud = GCloudSync()
        self.config = read_yaml_file(CONFIG_PATH)
        self.img_size = self.config['data_transformation_config']['img_size']

    def image_loader(self, image_bytes):
        """
        Method Name: Image loader
        Description: This method load byte image and save it to local
        Output: Returns path of the saved image        :
        """
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            logging.info("load byte image and save it to local")
            input_image = self.config['prediction_pipeline_config']['input_image']
            with open(input_image, 'wb') as image:
                image.write(image_bytes)
                image.close()
            path = os.path.join(os.getcwd(), input_image)
            image = Image.open(path)
            logging.info(f"Returns the saved image: {image}")
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return image
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_from_gcloud(self) -> str:
        """
        Method Name: Get_model_from_gcloud
        Description: This method fetched the best model from the gcloud
        Output: Return best model path
        """
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            logging.info("Loading the best model from gcloud bucket")
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel")
            self.gcloud.sync_file_from_gcloud(self.config['prediction_pipeline_config']["bucket_name"],
                                              self.config['prediction_pipeline_config']["model_name"],
                                              predict_model_path)
            best_model_path = os.path.join(predict_model_path, self.config['prediction_pipeline_config']["model_name"])
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def prediction(self, best_model_path: str, image) -> float:
        """
        Method Name: Prediction
        Description: This method takes the best model path and image
        Output: Return the image in base64
        """
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            logging.info("Loading best model")
            model = torch.load(best_model_path, map_location=DEVICE)
            model.eval()

            logging.info("Load the image and preprocess it")
            preprocess = transforms.Compose([
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.Grayscale(3),
                transforms.ToTensor()
            ])
            image = preprocess(image)
            image = image[:3]

            logging.info("Convert image to a PyTorch tensor and sent it to the device")
            image = image.unsqueeze(0).to(DEVICE)

            logging.info("Make the prediction")
            with torch.no_grad():
                logits = model(image)
                probs = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1)

            logging.info(f'Predicted label: {pred_label.item()}')
            logging.info("Map the predicted label to the corresponding class name")
            predicted_class_name = LABEL_NAME[pred_label.item()]
            logging.info(f'Predicted class name: {predicted_class_name}')
            logging.info("Exited the prediction method of PredictionPipeline class")
            return predicted_class_name

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            best_model_path: str = self.get_model_from_gcloud()
            detected_image = self.prediction(best_model_path, image)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image

        except Exception as e:
            raise CustomException(e, sys) from e

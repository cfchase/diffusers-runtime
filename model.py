import os
import io
import base64
import argparse
from PIL import Image

from kserve.protocol.rest.v2_datamodels import GenerateRequest
from typing import Dict, Union
import torch
from diffusers import DiffusionPipeline

from kserve import (
    Model,
    ModelServer,
    model_server,
    InferRequest,
    InferOutput,
    InferResponse,
)
from kserve.errors import InvalidInput
from kserve.utils.utils import generate_uuid

MODEL_ID = os.environ.get("MODEL_ID", default="stabilityai/stable-diffusion-2-1")
REFINER_ID = os.environ.get("REFINER_MODEL_ID", default="")

class DiffusersModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.pipeline = None
        self.refiner = None
        self.ready = False
        self.load()

    def load(self):
        pipeline = DiffusionPipeline.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        self.pipeline = pipeline
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def preprocess(
            self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Dict:
        if isinstance(payload, Dict) and "instances" in payload:
            # print("preprocess v1")
            headers["request-type"] = "v1"
        elif isinstance(payload, InferRequest):
            print("preprocess v2")
        else:
            raise InvalidInput("invalid payload")
        return {"prompt": payload["instances"][0]["prompt"]}

    def predict(
            self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:
        image = self.pipeline(payload["prompt"]).images[0]
        # image = Image.open(".ignore/img.png")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        im_b64 = base64.b64encode(image_bytes.read())

        if "request-type" in headers and headers["request-type"] == "v1":
            return {
                "predictions": [
                    {
                        "model_name": MODEL_ID,
                        "prompt": payload["prompt"],
                        "image": {
                            "format": "PNG",
                            "b64": im_b64
                        }
                    }
                ]}
        else:
            return {
                "model_name": MODEL_ID,
                "id": "id",
                "outputs": []
            }


parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = DiffusersModel(args.model_name)
    model.load()
    ModelServer().start([model])

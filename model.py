import os
import cv2
import numpy as np
import supervision as sv
from typing import List

import warnings
import requests
import random
import string
import torch

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model

# turn off pytorch warnings
warnings.filterwarnings("ignore") 

# set invironment
HOME = os.getcwd()
SAM_ENCODER_VERSION = "vit_h"
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")    
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize sam and grounding dino
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment_boxes_input(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def generate_sam_points(coords_tuple, label):
    sam_points, sam_labels = [], []
    sam_points.append([coords_tuple[0], coords_tuple[1]])
    sam_labels.append(bool(label == "yes"))
    return np.array(sam_points), np.array(sam_labels)

def segment_coords_input(sam_predictor: SamPredictor, image: np.ndarray, coords_tuple: tuple, label: str) -> np.ndarray:
    sam_predictor.set_image(image)
    point_coords, point_labels = generate_sam_points(coords_tuple, label)
    result_masks = []
    masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
    )
    index = np.argmax(scores)
    result_masks.append(masks[index])
    return np.array(result_masks)

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def download_image(image_url: str) -> str:
    image_format = image_url.split('.')[-1]
    data = requests.get(image_url).content
    filename = "{}.{}".format(get_random_string(8), image_format)
    filepath = os.path.join(HOME, "data", filename)    
    f = open(filepath,'wb')
    f.write(data)
    f.close()
    return filename, filepath
    
def segment_by_prompt(image_url: str, classes: list = ['gray pot', 'leafs']) -> dict:

    result = {}
    image_name, image_path = download_image(image_url)
    
    # should be parameters passed to the api
    # SOURCE_IMAGE_PATH = f"{HOME}/data/Week_3_plant_9_40_0.jpg"    
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    # load image
    image = cv2.imread(image_path)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=classes),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    # convert detections to masks
    detections.mask = segment_boxes_input(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    print(detections.xyxy)
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # write results
    outfile = os.path.join("/var/www/html", image_name)    
    cv2.imwrite(outfile, annotated_image)
    
    # print(detections)
    return {"mask": "test", "url": "http://ec2-3-253-15-150.eu-west-1.compute.amazonaws.com/{}".format(image_name)}

def segment_by_coords(image_url: str, coords_tuple: tuple, label: str) -> dict:
    result = {}
    image_name, image_path = download_image(image_url)
        
    # load image
    image = cv2.imread(image_path)
    
    detections = sv.Detections(xyxy=np.array([[0,0,1,1]]))
    
    # convert detections to masks
    detections.mask = segment_coords_input(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        coords_tuple=coords_tuple,
        label=label
    )

    # annotate image with detections
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    # write results
    outfile = os.path.join("/var/www/html", image_name)    
    cv2.imwrite(outfile, annotated_image)
    
    # print(detections)
    return {"mask": "test", "url": "http://ec2-3-253-15-150.eu-west-1.compute.amazonaws.com/{}".format(image_name)}


import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
from typing import Dict
import json
#from model.mvit import MViT

import traceback
import numpy as np
import torch
from tqdm import tqdm

import cv2
import utils.checkpoint as cu
import utils.logging as logging
import utils.misc as misc
from datasets import loader
from datasets import cv2_transform
from config.defaults import assert_and_infer_cfg
from utils.misc import launch_job
from utils.parser import load_config, parse_args
from copy import copy
import os

from model.build import build_model



####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
# Fix fillna
####
execute_in_docker = False

def get_sequence(center_idx, half_len, sample_rate, num_frames, length):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """

    if length % 2 == 0:
        seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))
    else:
        seq = list(range(center_idx - half_len, center_idx + half_len + 1, sample_rate))
    
    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1

    return seq

class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print(path)
        if not path.is_file():
            raise IOError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
            #cap = cv2.VideoCapture(str(fname))
        #return [{"video": cap, "path": fname}]
        return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )


class SurgVU_classify(ClassificationAlgorithm):
    def __init__(self, model, cfg):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-step-classification.json") if execute_in_docker else Path(
                "./output/surgical-step-classification.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        
        self.step_list = ["range_of_motion",
                          "rectal_artery_vein",
                          "retraction_collision_avoidance",
                          "skills_application",
                          "suspensory_ligaments",
                          "suturing",
                          "uterine_horn",
                          "other"]
        # Comment for docker build
        # Comment for docker built
        self.model = model
        self.cfg = cfg
        
        
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.ENDOVIS_DATASET.BGR

        self._crop_size = cfg.DATA.TEST_CROP_SIZE
        self._test_force_flip = cfg.ENDOVIS_DATASET.TEST_FORCE_FLIP
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP

        print(self.step_list)

    def dummy_step_prediction_model(self):
        random_step_prediction = random.randint(0, len(self.step_list)-1)

        return random_step_prediction
    
    def step_predict_json_sample(self):
        single_output_dict = {"frame_nr": 1, "surgical_step": None}
        return single_output_dict

    def process_case(self, *, idx, case):

        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # return
        # Write resulting candidates to result.json for this case
        return scored_candidates

    def save(self):
        print('Saving prediction results to ' + str(self._output_file))
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def _images_and_boxes_preprocessing_cv2(self, imgs):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = None

        try:
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
        except:
            breakpoint()
        
        imgs, boxes = cv2_transform.spatial_shift_crop_list(
            self._crop_size, imgs, 1, boxes=boxes
        )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        
        return imgs

    def _pack_pathway_output(self, frames):
        """
        Prepare output as a list of tensors. Each tensor corresponding to a
        unique pathway.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frame_list (list): list of tensors with the dimension of
                `channel` x `num frames` x `height` x `width`.
        """
        if self.cfg.DATA.REVERSE_INPUT_CHANNEL:
            frames = frames[[2, 1, 0], :, :, :]
        if self.cfg.MODEL.ARCH in self.cfg.MODEL.SINGLE_PATHWAY_ARCH:
            frame_list = [frames]
        elif self.cfg.MODEL.ARCH in self.cfg.MODEL.MULTI_PATHWAY_ARCH:
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.cfg.action_recognition.ALPHA
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
        else:
            raise NotImplementedError(
                "Model arch {} is not in {}".format(
                    self.cfg.MODEL.ARCH,
                    self.cfg.MODEL.SINGLE_PATHWAY_ARCH + self.cfg.MODEL.MULTI_PATHWAY_ARCH,
                )
            )
        return frame_list

    def predict(self, fname) -> Dict:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
 
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames == 0:
            return {}

        frame_indices = [i for i in range(num_frames)]

        window_size = 16
        sample_rate = 4

        for i in tqdm(frame_indices, desc=f'Processing video {fname}...'):
            frame_idx = 0
            window_frame_indices = get_sequence(i, (window_size * sample_rate) // 2, sample_rate, num_frames, window_size * sample_rate)

            frames = []

            for i, index in enumerate(window_frame_indices):
                # Set the current position of the video to the desired frame index
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)

                # Read the frame at the specified index
                ret, frame = cap.read()

                frames.append(frame)

            frames = self._images_and_boxes_preprocessing_cv2(frames)
            frames = self._pack_pathway_output(frames)

            frames[0] = frames[0].unsqueeze(0) #Agregamos la dimension del batch a nuestros datos

            print(frames[0].shape)

            #mvit_output = self.model(frames)




        ##
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        
        print('No. of frames: ', num_frames)

        # generate output json
        all_frames_predicted_outputs = []
        for i in range(num_frames):
            frame_dict = self.step_predict_json_sample()
            step_detection = self.dummy_step_prediction_model()

            frame_dict['frame_nr'] = i
            
            frame_dict["surgical_step"] = step_detection

            all_frames_predicted_outputs.append(frame_dict)

        steps = all_frames_predicted_outputs
        return steps

def test(cfg):
    """
    Perform testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """

    # Build the video model and print model statistics.
    model = build_model(cfg)
    print(model)

    # Load checkpoint #TODO: Revisar el script de checkpoint. Asegurarnos que el cfg tiene la ruta indicada de nuestro modelo final
    cu.load_test_checkpoint(cfg, model) 

    surgvu_pipeline = SurgVU_classify(model, cfg)

    # # Perform test on the entire dataset.
    with torch.no_grad():
        surgvu_pipeline.process()

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     SurgVU_classify().process()

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

####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
# Fix fillna
####
execute_in_docker = True

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
    def __init__(self):
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



if __name__ == "__main__":
    SurgVU_classify().process()

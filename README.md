# Docker image & algorithm submission for Category 2 of SurgVU Challenge 2024

This repository has everything you and your team need to make an algorithm submission for the [Surgical Visual Understanding Challenge](https://surgvu24.grand-challenge.org/) Category 2.

Be sure that you have a verified account on Grand Challenge and are accepted as a participant in the SurgVU challenge.
You should be able to submit your Docker container/algorithm on the challenge website when the submission opens.

Here are some useful documentation links for your submission process:
- [Tutorial on how to make an algorithm container on Grand Challenge](https://grand-challenge.org/blogs/create-an-algorithm/)
- [Docker documentation](https://docs.docker.com/)
- [Evalutils documentation](https://evalutils.readthedocs.io/)
- [Grand Challenge documentation](https://comic.github.io/grand-challenge.org/algorithms.html)

## Prerequisites

You will need to have [Docker](https://docs.docker.com/) installed on your system. We recommend using Linux with a Docker installation. If you are on Windows, please use [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install).

## Prediction format

For category 2 of [Surgical Visual Understanding Challenge 24](https://surgvu24.grand-challenge.org/) (surgical step classification) the instructions to generate the Docker container are given below:

### Category #2 – surgical step classification:  

The output json file needs to be a dictionary containing the set of surgical steps detected in each frame (as integers):

```
[
   {
      "frame_nr":0,
      "surgical_step":3
   },
   {
      "frame_nr":1,
      "surgical_step":4
   },
   {
      "frame_nr":2,
      "surgical_step":3
   },
   {
      "frame_nr":3,
      "surgical_step":0
   },
]
```

Below is the list of surgical steps (Integers are the index of the list, from 0 to 7):

```
step_list = ["range_of_motion",
            "rectal_artery_vein",
            "retraction_collision_avoidance",
            "skills_application",
            "suspensory_ligaments",
            "suturing",
            "uterine_horn",
            "other"]
```



## Adapting the container to your algorithm

TODO update this

1. First, clone this repository:

```
git clone https://github.com/aneeqzia-isi/surgtoolloc2022-category-2.git
```

2. Our `Dockerfile` should have everything you need, but you may change it to another base image/add your algorithm requirements if your algorithm requires it:

![Alt text](README_files/dockerfile_instructions.png?raw=true "Flow")

3. Edit `process.py` - this is the main step for adapting this repo for your model. This script will load your model and corresponding weights, perform inference on input videos one by one along with any required pre/post-processing, and return the predictions of surgical tool classification as a dictionary. The class Surgtoolloc_det contains the predict function. You should replace the dummy code in this function with the code for your inference algorihm. Use `__init__` to load your weights and/or perform any needed operation. We have added `TODO` on places which you would need to adapt for your model

4. Run `build.sh`  to build the container. 

5. In order to do local testing, you can edit and run `test.sh`. You will probably need to modify the script and parts of `process.py` to adapt for your local testing. The main thing that you can check is whether the output json being produced by your algorithm container at ./output/surgical-tools.json is similar to the sample json present in the main folder (also named surgical-tools.json).

 PLEASE NOTE: You will need to change the variable `execute_in_docker` to False while running directly locally. But will need to switch it back once you   are done testing, as the paths where data is kept and outputs are saved are modified based on this boolean. Be aware that the output of running test.sh, of course, initially may not be equal to the sample predictions we put there for our testing. Feel free to modify the test.sh based on your needs.

5. Run `export.sh`. This script will will produce `surgtoolloc_det.tar.gz` (you can change the name of your container by modifying the script). This is the file to be used when uploading the algorithm to Grand Challenge.

## Uploading your container to the grand-challenge platform

1. Create a new algorithm [here](https://surgtoolloc.grand-challenge.org/evaluation/challenge/algorithms/create/). Fill in the fields as specified on the form.

2. On the page of your new algorithm, go to `Containers` on the left menu and click `Upload a Container`. Now upload your `.tar.gz` file produced in step 5. 

3. After the Docker container is marked as `Ready`, you may be temped to try out your own algorithm when clicking `Try-out Algorithm` on the page of your algorithm. But doing so will likely fail. WARNING: Using this container in `Try-out` will fail. You can still use the Try-out feature to check logs from the algorithm and ensure that processes are running but it will not pass. However, if built correctly and you see the expected logs from your algorithm, then the container should still work for the Prelim submission. 

4. WE STRONGLY RECOMMEND that you make at least 1-2 Prelim submissions before the deadline to ensure that your container runs correctly. Start earlier (Aug 19th) so we can help debug issues that may arise, otherwise there will be no opportunities to debug containers during the main submission!

5. To make a submission to one of the test phases. Go to the [SurgToolLoc Challenge](https://surgtoolloc.grand-challenge.org/) and click `Submit`. Under `Algorithm`, choose the algorithm that you just created. Then hit `Save`. After the processing in the backend is done, your submission should show up on the leaderboard if there are no errors.

The figure below indicates the step-by-step of how to upload a container:

![Alt text](README_files/MICCAI_surgtoolloc_fig.png?raw=true "Flow")

If something does not work for you, please do not hesitate to [contact us](mailto:isi.challenges@intusurg.com) or [add a post in the forum](https://grand-challenge.org/forums/forum/endoscopic-surgical-tool-localization-using-tool-presence-labels-663/). 

## Acknowledgments

The repository is greatly inspired and adapted from [MIDOG reference algorithm](https://github.com/DeepPathology/MIDOG_reference_docker), [AIROGS reference algorithm](https://github.com/qurAI-amsterdam/airogs-example-algorithm) and [SLCN reference algorithm](https://github.com/metrics-lab/SLCN_challenge)

# surgvu2024-category2-submission

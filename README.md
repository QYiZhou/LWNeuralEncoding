# Code for paper “Exploring the brain-like properties of deep neural networks: a neural encoding perspective”

This repository contains the implementation of the layer-weighted ROI-wise encoding model described in **Exploring the brain-like properties of deep neural networks: a neural encoding perspective.**


## Dependencies
- python3.7
- torch == 1.7.0
- timm == 0.5.4
- numpy == 1.19.2
- skearn == 0.23.2
- csv == 1.0
- tensorboardX == 2.1

## Usage
**Step 1: Dataset preparation**

The dataset can be downloaded from [The Algonauts Project 2021](http://algonauts.csail.mit.edu/challenge.html). Save stimuli and neural responses to ```./AlgonautsVideos268_All_30fpsmax``` and ```./participants_data_v2021```, respectively.


**Step 2: Feature extraction** 

Only AlexNet. The function of extracting features of different DNNs is coming soon.

**Step 3: Model fitting**

Train the layer-weighted ROI-wise encoding model. First, select best hyper-parameters on the first subject. Second, train the model using the selected hyper-parameters on all subjects and all ROIs.

**Step 4: Result summarizing**

Save the average results on all subjects into a ```.csv``` file.

You can also run step 2-4 by ```bash ./code/run.sh``` directly. The results will be saved in ```./results```.

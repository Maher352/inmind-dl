# InMindAcademy
This repository is used to upload all the python files developed in the academy

The main purpose of this repository is to get familiar with git commands and GitHub

The second purpose of this repository is project submission

!! yolov5 must be cloned to the repository as it is used for object detection


Regarding the project many things can be done to enhance it:

1- the dataloader developped for the yolov5 model should receive the directories of images and npy files with the transform and output 
    a- a folder containing the augmented images split into train and validate
    b- with a folder containing text files ready to use by yolov5 (also split)
    c- build the yaml file needed for the yolov5 training
This will allow quick usage before training the yolov5 model

2- the semantic segmentation model is not complete due to deadline restrictions

3- the onnx inference should be fixed to output a clear and CORRECT prediction image

4- API inference can be added to make the user experience more user-friendly and efficient

5- these can be added: 
    a- Ensemble Models: Use ensemble methods to combine the results of multiple 
        object detection or segmentation models to improve accuracy.
    b- Transfer Learning: Apply transfer learning from a pre-trained model on a related 
        dataset and fine-tune it on the given dataset.
    c- Real-time Inference: Implement a real-time inference pipeline that processes 
        live video feeds and displays the results
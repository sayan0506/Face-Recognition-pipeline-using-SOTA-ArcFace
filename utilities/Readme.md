This folder contains the utility files to run the ML pipeline

Order of execution for the training pipeline

1. [train.py](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/utilities/train.py) - runs entire train pipeline
2. [FaceDataset.py](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/utilities/FaceDataset.py) - implements custom torch dataset class to create train, valid, test dataset, dataloader
3. [model.py](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/utilities/model.py) - defines mobilefacenet, ir-se50, arcface head, arcface softmax
4. [face_learner.py](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/utilities/face_learner.py) - implements the pre-trained model initialization, load pre-trained weights, save state dict, load state dict..
5. [config.yml](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/utilities/config.yml) - records all the training configuration

Order of execution for the test pipeline

1. [test.py]() - implements the test pipeline
2. [FaceDataset.py]() - implements custom torch dataset class to create valid, test dataset, dataloader 
3. [inference_learner.py]() - implements the trained model initialization, load best trained weights..

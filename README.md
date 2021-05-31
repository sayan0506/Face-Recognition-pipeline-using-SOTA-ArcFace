# Face-Recognition-pipeline-using-SOTA-ArcFace
Face Recognition pipeline using SOTA ArcFace using Pytorch

# **Problem Statement**

The problem statement for face recognition task can be found [SkylarkLabs-ai-link](https://www.notion.so/SkylarkLabs-ai-894729b2086c4136bcc751298cada7a3)

# **Dataset**

The folders contain around 50 cropped face images of each avenger can be found from [kaggle Avengers face recognition!](https://www.kaggle.com/rawatjitesh/avengers-face-recognition)

1. Chris Evans (Captain America)
2. Chris Hemsworth (Thor)
3. Mark Ruffalo (Hulk)
4. Robert DowneyJr (The Iron man)
5. Scarlett Johansson (Black Widow)

The dataset zip file or folder can be found from drive [link](https://drive.google.com/drive/folders/1VYuEXVOzUtd7fOaaLv7oW4YwGYbvh80w?usp=sharing)

# **Notebook**

The detailed code implementation for the ArcFace implementation using pytorch can be found from [colab notebook](https://colab.research.google.com/drive/1l3HfvGbvRxlV2XxZ_9gidsQdvPui1JcS?usp=sharing)

# **Best Model**

* We have choosen [MobileFacenet model](https://arxiv.org/abs/1804.07573), which is a lightweight model as compared to the vailable pre-trained [IR-SR50 model](https://arxiv.org/abs/1801.07698), 
* We have done transfer learning using pre-trained mobile-facenet model trained on MS1M, VGG2, Emore dataset. 
* The trained best model evaluated on validation data(lowest validation loss) can be foudn [here](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/model_weights)
* More on the models, implementation can be found [InsightFace_Pytorch - Arcface Pytorch Implementation github](https://github.com/TreB1eN/InsightFace_Pytorch) 

## **ML Pipeline**

The train.py, test.py and related utilities to execute train, test pipeline can be found from **[utilities folder](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/utilities)**

#### **Guide to run**

Execute training

```bash
    python train.py -dataset [dataset_path] -model [model folder]
```   
    
# **Reference**

* [How to train a deep learning model using docker?](https://www.youtube.com/watch?v=Kzrfw-tAZew)
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition - paperwithcode-reference](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep)
* [InsightFace_Pytorch - Arcface Pytorch Implementation github](https://github.com/TreB1eN/InsightFace_Pytorch)
* [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
* [Docker Documentation](https://docs.docker.com/)
* [Docker For Data Scientists](https://www.youtube.com/watch?v=0qG_0CPQhpg)

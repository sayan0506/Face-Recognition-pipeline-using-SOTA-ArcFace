# Face-Recognition-pipeline-using-SOTA-ArcFace

Face Recognition pipeline using SOTA ArcFace using Pytorch is inspired from the paper [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698v3.pdf)

Git repo: [Insightface_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)

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

## **Guide to run**

1. Download and unzip the dataset from [here](https://www.kaggle.com/rawatjitesh/avengers-face-recognition)

2. Git clone the repo - [repo_name](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace)

```bash
    git clone [repo_name]
```   
3. Navigate to repo to install dependencies from - [requirements.txt](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/blob/main/requirements.txt)

```bash
    cd Face-Recognition-pipeline-using-SOTA-ArcFace
    pip install -r requirements.txt
```   
4. Navigate to **"utilities folder"** which contains dependencies to run the pipeline - [utilities_path](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/utilities)

```bash
    cd [utilities_path]
```   
5. Execute train pipeline - [pre-trained-mobilefacenet model folder](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/model_weights). The train_pipeline code can be tweaked little to change the pre-trained model file.

```bash
    python train.py -dataset [dataset_path] -model [pre-trained-mobilefacenet model folder]
```   
6. Execute test pipeline - 
* [trained_mode_folder](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/model_weights). The train_pipeline code can be tweaked little to change the pre-trained model file.
* [csv_folder_demo](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/demo_csv), the path where the train, valid, test csv files are stored from the dataframe

```bash
    python test.py -dataset [dataset_path] -model [trained_mode_folder] -df [csv_folder]
```   

* A simple demo of that train, test pipeline can be found [here](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/test_demo).
* Docker implementation in colab can be found [here](https://github.com/sayan0506/Face-Recognition-pipeline-using-SOTA-ArcFace/tree/main/Docker%20tests), though it needs more testing to execute the layers.

# **Reference**

* [How to train a deep learning model using docker?](https://www.youtube.com/watch?v=Kzrfw-tAZew)
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition - paperwithcode-reference](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep)
* [InsightFace_Pytorch - Arcface Pytorch Implementation github](https://github.com/TreB1eN/InsightFace_Pytorch)
* [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
* [Docker Documentation](https://docs.docker.com/)
* [Docker For Data Scientists](https://www.youtube.com/watch?v=0qG_0CPQhpg)
* [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)
* [Additive Margin Softmax Loss (AM-Softmax)](https://towardsdatascience.com/additive-margin-softmax-loss-am-softmax-912e11ce1c6b)
* [The difference between Softmax and Softmax-Loss](https://medium.com/@liangjinzhenggoon/the-difference-between-softmax-and-softmax-loss-173d385120c2)

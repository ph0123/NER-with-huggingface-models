# NER-with-huggingface-models

- In this example, I run the models with CPU only and randomly select a small part of the data for training and testing. The ratio of the training set : validation set: test set is ~ 4:3:3. It took around 100 minutes to train 10 epochs.

- Computer: Dell, core i7, 64GB RAM.
- Pre-trained Model: [Distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert). This is because of the limitations of my computer.
  
`"DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark."`


## Findings
- The testing set has missed a label ('I-BIO' - there are only 30 labels) compared to the training set (31) and validation set (31).
- Time for running: 10 epochs for training: 
  - SystemA: 5619.6813 s ~ 1.56 hour. (31 Labels)
  - SystemB: 5557,2894 ~ 1.5436915. (11 labels with five man labels: PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS),
ANIMAL(ANIM)).
  - `The times for training of 2 systems are not far different. This is because there are changes in the output layer in the models. Other layers are trained in similar ways. However, it can be different with entire datasets because I only test these systems on small random data from the original data because of time limitations and my machine.
`
- Accuaracy (Precision | recall | f1 score)
  - System A:	  1.00	 |  0.83 	| 0.88
  - System B:   0.99 |	0.90	|	0.91
  - System B outperformed System A on average score. This is because of fewer labels compared to system A. 

- Limitations: 
  - Randomly small parts of the data can lead to incorrect evaluation of the models. Therefore, It needs to run with a powerful computer for all datasets. 
  - Because of time limitations, I only tested on the default learning rate (example) from HuggingFace. It can change the learning rate with different optimizers like Adam. (See more here)
  - Epoch = 10. It needs to be tested with more epochs. Note that I run one epoch with the entire training set. It took more than 8 hours. According to the system, you can consider how many epochs you need for training. GPU is another solution to reduce the time for training. 


## Software requirements:
- python>=3.9
- You can see it [requirements.txt](./requirements.txt).

    pip install -r requirements.txt

## Installations:

- Install Anaconda [Here](https://docs.anaconda.com/free/anaconda/install/index.html). My conda version is 22.11.1
- Create an environment with anaconda: `conda create -n MY_ENV python=3.9`
- Go to the environment: `conda activate MY_ENV`
- Install pip (Usually, pip is installed when creating MY_ENV): `conda install pip` 
- Install Jupyter Notebook: `conda install -c conda-forge notebook`
- Install additional packages: `pip install -r requirements.txt`

## Datasets:
- See Data folder [Eng-MultiNERD](https://drive.google.com/drive/folders/1MvEsk6eiayAnWzAejcNYrEHXVVtbKUH1).

### Note 1: You need to check the data directory to run the codes. 
- For me, I put the datasets in the same folder with the codes.
- If you want to run SystemA and SystemB with my models. You can download it from [My trained models](https://drive.google.com/drive/folders/1P22n3j08eAsyvuBZU63mNEMdK9Dqj6et?usp=sharing).

## Usage:
- Clone the repository, download the datasets, and pre-trained models (If you do not want to re-train the models) locally and run it step by step.
- Note: for the first cell with `"!pip install ...."`. Uncomment this cell if you have not already installed these libraries.
  
1. Run all cells in [DataStatistic.ipynb](./DataStatistic.ipynb) to check the data information.
   
   - Sizes of the training set, validation set and test set are 131280, 16410, and 16454, respectively. 
   - Sizes of small samples with training set, validation set and test set are 3282, 2461, and 2468, respectively.
2. You need to run if you want to train the model from DistillBert and evaluate the models. If not, you can skip this step and go to step 3. 
   
   2.1 Run all cells for [SystemADistilbert.ipynb](./SystemADistilbert.ipynb). You will get the "Best model Distillbert after 10 epochs - SystemA" folder, which can be downloaded from [My trained models](https://drive.google.com/drive/folders/1P22n3j08eAsyvuBZU63mNEMdK9Dqj6et?usp=sharing).

   2.2 Run all cells for [SystemBDistilbert.ipynb](./SystemBDistilbert.ipynb). You will get the "Best model Distillbert after 10 epochs - SystemB" folder, which can be downloaded from [My trained models](https://drive.google.com/drive/folders/1P22n3j08eAsyvuBZU63mNEMdK9Dqj6et?usp=sharing).

3. Load and Test System A and B with an example sentence.
- Sentences: `"Denmark's	Hans Nielsen won his third World Championship with a 15-point maximum from his five rides."`
- Run all cells of [SystemADistilbert_Testing.ipynb](./SystemADistilbert_Testing.ipynb) to see how the trained model works for SystemA. 
- Run all cells of [SystemBDistilbert_Testing.ipynb](./SystemBDistilbert_Testing.ipynb) to see how the trained model works for SystemB. 

## Note2

- You can read the [slides](./Presentations_of_My_Examples.pptx) to understand more details and the results of these systems in this repository.

Cheers!
    
Chau

cnphuongdavid@gmail.com

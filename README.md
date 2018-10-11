# Final Pipeline Instructions
## Setup
In a command line:
1. Make sure Python 3.x is installed
3. If using a Mac make sure Home Brew is installed
2. run `setup.sh`

## Pipeline Overview
This pipeline works in two main steps. The first step is the optical character recognition, which takes an input direct within the command line. The input directory is the directory that contains the files you would like to be classified within our system. These files can consist of many different types. After the OCR is ran, a hidden output directory called `intermediate_directory` will hold a copy of the original files that have been converted into text files. This is done because our classification system can only handle text documents. 

The second main step is the classification of the different files based on the personally identifiable information within them. Our system will take in the hidden directory that contains the text files, and predict the types of personally identifiable information within them. From here, the original files will be placed within directories that relate to these categories: Non Personal, Personal, and Sensitive Personal. Within each subdirectory, there will be two overview documents that explains which types of PII each document contains. 

## How to Run
### Full
In command line run:
- `./classifier full input_directory output_directory`

### OCR
In command line run:
- `./classifier ocr input_directory intermediate_directory` 

### Classification
In command line run:
- `./classifier classify model_zip_file intermediate_directory output_directory`


# Training/ Data Labeling Instructions
If you would like to train a model further train your own model, then more data labeling and model training has to occur. This can be done within our system, and has to be done on the text versions of the documents. If you would like to find these text versions, after running the OCR on new training data find `intermediate_directory` within your file system. 

## Document Level Labeling
Document level labeling is used to train the models that classify documents into the  Non-Personal, Personal and Sensitive Personal Data categories. To label more training data:
1. Create three directories, one for each: Non-Personal, Personal, Sensitive Personal data
2. Identify category of PII and copy new training data into the corresponding directory

## Training Model
After Labeling more data run:
- `./classifier train new_data`

This will output a zip file containing the weights of the new model




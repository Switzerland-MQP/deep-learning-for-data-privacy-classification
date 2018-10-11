'''
Run this on text files to replace common misspellings & remove stopwords found within our documents
'''
import os


def clean_arrows(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = root + '/' + file
            
            with open(full_path, 'r') as f:
                filedata = f.read()
            
            filedata = filedata.replace('>', '~')
            filedata = filedata.replace('<', '~')
            
            with open(full_path, 'w') as f:
                f.write(filedata)

#  clean_arrows('./arrows')

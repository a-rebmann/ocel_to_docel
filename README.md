# OCEL to DOCEL Datasets and Automated Transformation
Tools to generate DOCEL formatted event logs and implementation of an algorithm to transform OCEL into DOCEL logs

## Setup 
### Requirements
- Python 3.9
- install requirements.txt (pip install -r requirements.txt)
- We use a custom label tagger to extract objects, actions, and various other semantic information from labels. 
You need to download the four files from [here](https://gitlab.uni-mannheim.de/processanalytics/semantic-event-log-annotation/-/tree/main/.model/main) and put them into model/
- we use a Spacy model to extract the objects from the labels. To install it run `python -m spacy download en_core_web_sm`
### Usage
run main.py after placing the OCEL file into the input folder. The output will be placed in the output folder.

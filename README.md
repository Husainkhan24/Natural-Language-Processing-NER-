# Natural-Language-Processing-NER-
Clinical Text Classification Using NLP technique Name Entity Recognition

Natural language processing (NLP) can be used to answer a variety of questions about unstructured text, as well as facilitating open-ended exploration. It can be applied to datasets such as emails, online articles and comments, tweets and novels. Although the source is text, transformations are applied to convert this data to vectors, dictionaries and symbols which can be handled very effectively by q. Many operations such as searching, clustering, and keyword extraction can all be done using very simple data structures, such as feature vectors.

## Dependancies
import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
import json 
import pickle

#Dataset had taken from kaggle https://www.kaggle.com/datasets/ner-dataset
DATA

{'examples': [{'id': '18c2f619-f102-452f-ab81-d26f7e283ffe',
   'content': "While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] These agents should be used only if bloody diarrhea is not present.[92]\n\nDiosmectite, a natural aluminomagnesium silicate clay, is effective in alleviating symptoms of acute diarrhea in children,[93] and also has some effects in chronic functional diarrhea, radiation-induced diarrhea, and chemotherapy-induced diarrhea.[45] Another absorbent agent used for the treatment of mild diarrhea is kaopectate.\n\nRacecadotril an antisecretory medication may be used to treat diarrhea in children and adults.[86] It has better tolerability than loperamide, as it causes less constipation and flatulence.[94]",
   'metadata': {},
   'annotations': [{'id': '0825a1bf-6a6e-4fa2-be77-8d104701eaed',
     'tag_id': 'c06bd022-6ded-44a5-8d90-f17685bb85a1',
     'end': 371,
     'start': 360,
     'example_id': '18c2f619-f102-452f-ab81-d26f7e283ffe',
     'tag_name': 'Medicine',
     'value': 'Diosmectite',
     'correct': None,
     'human_annotations': [{'timestamp': '2020-03-21T00:24:32.098000Z',
       'annotator_id': 1,
       'tagged_token_id': '0825a1bf-6a6e-4fa2-be77-8d104701eaed',
       'name': 'Ashpat123',
       'reason': 'exploration'}],
     'model_annotations': []}
     
 ### Data Preprocessing
 
#spacy training data format https://spacy.io/usage/training#training-data
#['tag_name'],['start'],['end'] 
#DATA['examples'][0]['annotations'][0]['start'] 
len(DATA['examples'])  [31]
pre_data=[]
for i in range(0,len(DATA['examples'])):
    content=DATA['examples'][i]['content'] 
    #content=content.replace("."," ").replace(","," ").replace("["," ").replace("]"," ")
    #content=content.replace("%"," ").replace("â"," ").replace("€"," ").replace("“"," ").replace('\n',' ')
    annot=DATA['examples'][i]['annotations']
    fex=[]
    for j in annot:
        s=j['start']
        e=j['end']
        l=j['tag_name']
        fex.append((s,e,l))
   #print(DATA['examples'][i]['annotations'])
    pre_data.append((content,{'entities':fex}))
    

('Antiretroviral therapy (ART) is recommended for all HIV-infected individuals to reduce the risk of disease progression.\nART also is recommended for HIV-infected individuals for the prevention of transmission of HIV.\nPatients starting ART should be willing and able to commit to treatment and understand the benefits and risks of therapy and the importance of adherence. Patients may choose to postpone therapy, and providers, on a case-by-case basis, may elect to defer therapy on the basis of clinical and/or psychosocial factors.',
  {'entities': [(0, 22, 'Medicine'),
    (24, 27, 'Medicine'),
    (120, 123, 'Medicine'),
    (211, 214, 'Pathogen'),
    (52, 55, 'Pathogen'),
    (234, 237, 'Medicine'),
    (148, 151, 'Pathogen')]})    
    
    
### spacy Data format  
nlp=spacy.blank('en')
doc_bin=DocBin()
for train_eg in tqdm(pre_data[0:26]):
  text=train_eg[0]
  labels=train_eg[1]['entities']
  doc=nlp.make_doc(text)
  ents=[]
  for start,end,label in labels:
    span=doc.char_span(start,end,label=label)
    if span is None:
     pass#print("skipping Entity")
    else:
      ents.append(span)
  filtered_ents=filter_spans(ents)
  doc.ents=filtered_ents
  doc_bin.add(doc)
doc_bin.to_disk('/content/drive/MyDrive/github_proj/TRAIN_Davy.spacy')
    
   
   
*!python -m spacy download en_core_web_lg

✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_lg')

*!python -m spacy init fill-config /content/drive/MyDrive/github_proj/base_config.cfg /content/drive/MyDrive/github_proj/config.cfg

✔ Auto-filled config with all values
✔ Saved config

# Training
*!python -m spacy train /content/drive/MyDrive/github_proj/config.cfg --output  /content/drive/MyDrive/github_proj/ --paths.train /content/drive/MyDrive/github_proj/TRAIN_Davy.spacy  --paths.dev /content/drive/MyDrive/github_proj/TEST_Davy.spacy

ℹ Pipeline: ['tok2vec', 'ner']
ℹ Initial learn rate: 0.001
E    #       LOSS TOK2VEC  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
---  ------  ------------  --------  ------  ------  ------  ------
  0       0          0.00    143.29    0.00    0.00    0.00    0.00
  1      50        160.63   1340.59    0.00    0.00    0.00    0.00
  3     100         30.32    700.46   38.60   47.83   32.35    0.39
  5     150          6.71    463.10   61.54   64.52   58.82    0.62
  7     200         13.19    462.49   46.15   48.39   44.12    0.46
  9     250         39.11    401.21   53.16   46.67   61.76    0.53
 11     300        370.06    517.07   65.62   70.00   61.76    0.66
 13     350         19.12    192.57   51.35   47.50   55.88    0.51
 15     400         11.01    117.73   58.62   70.83   50.00    0.59
 17     450         54.36    150.26   49.12   60.87   41.18    0.49
 19     500         15.89    136.84   60.00   69.23   52.94    0.60
 21     550         15.05     95.32   52.83   73.68   41.18    0.53
 23     600         13.19     57.53   50.00   53.33   47.06    0.50
 25     650         17.22     76.50   50.70   48.65   52.94    0.51
 26     700         11.21     47.41   55.88   55.88   55.88    0.56
 28     750         18.64     64.91   48.15   65.00   38.24    0.48

import json
import spacy
import re
import pandas as pd

from nltk.corpus import stopwords
from tqdm import tqdm
from collections import deque

nlp = spacy.load("en_core_web_sm")


def clean(label):
    # handle some special cases
    label = label.replace("\n", " ").replace("\r", "")
    label = label.replace("(s)", "s")
    # turn any non alphanumeric characters into whitespace
    label = re.sub("[^A-Za-z]"," ",label)
    # delete unnecessary whitespaces
    label = label.strip()
    label = re.sub("\s{1,}"," ",label)
    # make all lower case
    label = label.lower()
    return label

def alphanumeric(graph_name):
    return re.sub("[^A-Za-z]"," ",graph_name)

def get_bpmn_event_labels(model_json_str, eventTypes):
    model_json = json.loads(model_json_str)
    model_labels = ""
    stack = deque([model_json])
    while len(stack) > 0:
        element = stack.pop()
        try:
            if "stencil" in element:
                if(element["stencil"]["id"] in eventTypes):
                        if "name" in element["properties"]:
                            if(element["properties"]["name"]!=""):
                                title = isNotStopword(element["properties"]["name"]) #In BPMN, the name of an element is called name
                                model_labels += clean(title)+", ";
            for c in element.get("childShapes", []):
                stack.append(c)
        except Exception as error:
            print("Error: "+str(error))
    return model_labels

def get_epc_pm_event_labels(model_json_str, eventTypes):
    model_json = json.loads(model_json_str)
    model_labels = ""
    stack = deque([model_json])
    while len(stack) > 0:
        element = stack.pop()
        try:
            if "stencil" in element:
                for childShapes in element["childShapes"]:
                    if childShapes['stencil']['id'] in eventTypes:
                        if 'title' in childShapes['properties']:
                            if(childShapes["properties"]["title"]!=""):
                                title = isNotStopword(childShapes["properties"]["title"]) #In EPC, the JSON structure is different. The name of an element is called title
                                model_labels += clean(title)+", ";
            for c in element.get("childShapes", []):
                stack.append(c)
        except Exception as error:
            print("Error: "+str(error))
    return model_labels

def isNotVerbAndStopword(text):
    doc = nlp(text) #POS analysis, e.g. [('This', 'PRON'), ('is', 'AUX'), ('a', 'DET'), ('sentence', 'NOUN'), ('.', 'PUNCT')]
    filteredText = [str(token) for token in doc if((token.pos_ != "VERB") and (str(token) not in stopwords.words('english')))]
    filteredText = ' '.join(filteredText)
    return filteredText

def isNotStopword(text):
    doc = nlp(text)
    filteredText = [str(token) for token in doc if(str(token) not in stopwords.words('english'))]
    filteredText = ' '.join(filteredText)
    return filteredText

def returnElementTypes(notation):
    with open("d_types_mapping.json") as f:
        d_types_mapping = json.load(f)
        d_low_level = d_types_mapping[notation]
    eventTypes = []
    for element in d_low_level:
        if(d_low_level[element] == "event" or d_low_level[element] == "task" or d_low_level[element] == "lane" or d_low_level[element] == "pool"):
            eventTypes.append(element)
    return eventTypes;

def get_event_tags_and_model_names_for_data_frame(df):
    eventTypesBPMN2=returnElementTypes("d_types_bpmn2")
    eventTypesBPMN1=returnElementTypes("d_types_bpmn1")
    eventTypesEPC=returnElementTypes("d_types_EPC")
    eventTypesPM=returnElementTypes("d_types_process_map")
    model_id_list=list()
    event_labels_list=list()
    model_names_list=list()
    for index, value in tqdm(df.iterrows()):
        model_id_list.append(value['Model ID'])
        if(value['Namespace']=='http://b3mn.org/stencilset/bpmn1.1#'):
            event_labels_list.append(get_bpmn_event_labels(value['Model JSON'], eventTypesBPMN1))
        if(value['Namespace']=='http://b3mn.org/stencilset/bpmn2.0#'):
            event_labels_list.append(get_bpmn_event_labels(value['Model JSON'], eventTypesBPMN2))
        if(value['Namespace']=='http://b3mn.org/stencilset/epc#'):
            event_labels_list.append(get_epc_pm_event_labels(value['Model JSON'], eventTypesEPC))
        elif(value['Namespace']=='http://www.signavio.com/stencilsets/processmap#'):
            event_labels_list.append(get_epc_pm_event_labels(value['Model JSON'], eventTypesPM))
        model_names_list.append(value['Name'])
    zipped = list(zip(model_id_list, event_labels_list, model_names_list))
    event_labels_df = pd.DataFrame(zipped, columns=["model_id", "event_labels", "model_names"]).reset_index(drop=True) #.reset_index(drop=True) was added newly, did not check if working
    return event_labels_df

def add_event_labels_to_DF(namespaceFiltered_non_demo_models_df):
    namespace_event_labels_and_model_names_df = get_event_tags_and_model_names_for_data_frame(namespaceFiltered_non_demo_models_df)
    #merge:
    namespaceFiltered_non_demo_models_df.rename(columns={"Model ID":"model_id"}, inplace=True)
    namespaceFiltered_non_demo_models_df['model_id']=namespaceFiltered_non_demo_models_df['model_id'].astype(str)
    namespace_event_labels_and_model_names_df['model_id']=namespace_event_labels_and_model_names_df['model_id'].astype(str)
    namespace_filtered_non_demo_event_labels_models=pd.merge(namespaceFiltered_non_demo_models_df, namespace_event_labels_and_model_names_df.drop(columns=['model_names']), on='model_id', how='left')
    return namespace_filtered_non_demo_event_labels_models
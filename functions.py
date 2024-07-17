import pandas as pd
import io
import os
import json
import requests

from PIL import Image
import pytesseract
import easyocr

from langchain.output_parsers import PydanticOutputParser

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# import time
import torch

from tqdm.auto import tqdm

import constants as const
from FactureModel import Facture


def readDataSet(bigDataset: bool):
    # Read the dataset on pandas and join train and test
    if bigDataset:
        splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
        df_train = pd.read_parquet("hf://datasets/doceoSoftware/docvqa_clicars_facturavenda_Mireia_600_3/" + splits["train"])
        df_test = pd.read_parquet("hf://datasets/doceoSoftware/docvqa_clicars_facturavenda_Mireia_600_3/" + splits["test"])
    else:
        import pandas as pd
        splits = {'train': 'data/train-00000-of-00001-45c8bfebf3cf5109.parquet', 'test': 'data/test-00000-of-00001-02214854a42c16ee.parquet'}
        df_train = pd.read_parquet("hf://datasets/ayoub999/dataset_for_orange_factures/" + splits["train"])
        df_test = pd.read_parquet("hf://datasets/ayoub999/dataset_for_orange_factures/" + splits["test"])

    return df_train._append(df_test, ignore_index=True) 

def imageObjCreation(image_bytes):
    image_file = io.BytesIO(image_bytes)
    # Open the image using PIL
    return Image.open(image_file)



def readImgOCR(image_bytes, image, doEasyOCR:bool):
    if not doEasyOCR:
        text = pytesseract.image_to_string(image, lang='eng')

        # Mostrar el texto extraído y los datos de la factura
        print("Texto extraído con OCR:")
        print(text)
        return text

    else:
        # Inicializar el lector de EasyOCR
        reader = easyocr.Reader(['en', 'es', 'fr'])  # Puedes agregar más idiomas en la lista si es necesario

        # Realizar OCR
        textEasyOcr = reader.readtext(image_bytes,detail=0)
        textEasyOcr = " ".join(textEasyOcr)

        # Mostrar el texto extraído y los datos de la factura
        print("Texto extraído con OCR:")
        print(textEasyOcr)
        return textEasyOcr


def pydanticParser():
    return PydanticOutputParser(pydantic_object = Facture)


def instructionsFormat(parser, textOcr):
    
    system_instructions = const.system_template.format(
        format_instructions=parser.get_format_instructions(),
    )

    prompt = const.prompt_template.format(
        data=textOcr,
    )

    return system_instructions, prompt


def LLMModelCall(system_instructions, prompt):
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYTZmNGZlZTUtZTU2ZS00NGU1LTliZTktMmExNTA3ZWFhMWQ5IiwidHlwZSI6ImFwaV90b2tlbiJ9.Bsw1LC2PXi2WND5Uqxh4CxIw-gD85ncDXDIrQgnN5I4"}

    #files = str({'file': ('image.png', df["image"][13]["bytes"])})

    url = "https://api.edenai.run/v2/text/chat"
    payload = {
        "providers": "openai",
        "text": prompt,
        "chatbot_global_action": system_instructions,
        "previous_history": [],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    response = requests.post(url, json=payload, headers=headers)

    # Asegúrate de manejar correctamente la respuesta para extraer los datos de la factura
    result = response.json()
    return result['openai']['generated_text']


def embedingModelCreation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('Sorry no cuda.')
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)


def dataBaseCreation(model):
    pinecone = Pinecone(api_key=const.PINECONE_API_KEY)
    INDEX_NAME = "quickstart" 

    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    print(INDEX_NAME)
    pinecone.create_index(name=INDEX_NAME, 
        dimension=model.get_sentence_embedding_dimension(), 
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

    index = pinecone.Index(INDEX_NAME)
    print(index)
    return index



def insertDataBase (index, l_generated_text, model):
    for i in tqdm(range(0, len(l_generated_text))):

        ids = [str(i)]
        metadatas = [json.loads(l_generated_text[i])]
        xc = [model.encode(l_generated_text[i])]

        # create records list for upsert
        records = zip(ids, xc, metadatas)
        # upsert to Pinecone
        index.upsert(vectors=records)



# small helper function so we can repeat queries later
def runQuery(query, index, model):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']}")



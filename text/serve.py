#!/usr/bin/env python

from transformers import BertModel, BertTokenizerFast
from flask import Flask, Response, request
import transformers
import logging
import torch
import nltk
import sys
import os


app = Flask(__name__)

DATA_PATH = '/opt/ml/model'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TextEncoder:
    
    bert_model = None
    bert_tokenizer = None
    nltk_tokenizer = None
    
    @classmethod
    def load_bert_model(cls):
        if cls.bert_model is None:
            cls.bert_model = BertModel.from_pretrained(f'{DATA_PATH}/bert-model/')
        return cls.bert_model
    
    @classmethod
    def load_bert_tokenizer(cls):
        if cls.bert_tokenizer is None:
            cls.bert_tokenizer = BertTokenizerFast.from_pretrained(f'{DATA_PATH}/bert-tokenizer/')
        return cls.bert_tokenizer
    
    @classmethod
    def load_nltk_tokenizer(cls):
        if cls.nltk_tokenizer is None:
            cls.nltk_tokenizer = nltk.load(f'{DATA_PATH}/english.pickle')
        return cls.nltk_tokenizer
            
    @classmethod 
    def encode(cls, text):
        bert_model = cls.load_bert_model()
        bert_tokenizer = cls.load_bert_tokenizer()
        nltk_tokenizer = cls.load_nltk_tokenizer()
        sentences = nltk_tokenizer.tokenize(text)
        
        max_len = 0
        for sentence in sentences:
            input_ids = bert_tokenizer.encode(sentence, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        
        input_ids = []
        attention_masks = []
        
        for sentence in sentences:
            encoded_dict = bert_tokenizer.encode_plus(sentence, 
                                                      add_special_tokens=True, 
                                                      max_length=max_len, 
                                                      padding='max_length', 
                                                      return_attention_mask=True, 
                                                      return_tensors='pt', 
                                                      truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_masks)
        
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        sentence_vectors = last_hidden_state.detach().numpy()
        paragraph_vector = sentence_vectors.mean(axis=0)  # mean of each column
        paragraph_vector = paragraph_vector.tolist()
        return paragraph_vector
    
@app.route('/ping', methods=['GET'])
def ping():
    status = 200
    return Response(response={'HEALTH CHECK': 'OK'}, status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invoke():
    if request.content_type == 'text/csv':
        payload = request.data.decode('utf-8')
    else:
        return flask.Response(response='This encoder only supports CSV data', status=415, mimetype='text/plain')
        
    logger.info(f'Incoming payload: {payload}')
    
    encoded_data = TextEncoder.encode(payload)
    feature_vector = ','.join(map(str, encoded_data))
    
    logger.info(f'Feature vector: {feature_vector}')

    return Response(response=feature_vector, status=200, mimetype='text/csv')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
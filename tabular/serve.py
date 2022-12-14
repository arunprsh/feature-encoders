#!/usr/bin/env python

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, Response, request
from datetime import datetime
from itertools import chain
from io import StringIO
import pandas as pd
import numpy as np
import logging
import pickle
import joblib
import os


app = Flask(__name__)

DATA_PATH = '/opt/ml/model'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FeatureEncoder:
    """
    Singleton Class for holding the transforms
    """
    date_transformer = None
    minmax_scaler_total_purchases = None
    minmax_scaler_total_reviews = None
    minmax_scaler_purchases_last_60_days = None
    minmax_scaler_reviews_last_60_days = None
    onehot_encoder = None
    age_group_encoder = None
    
    MAX_ACTIVE_DAYS = 10024
    MAX_AGE_GROUP = 5
    
    @classmethod
    def load_date_transformer(cls):
        if cls.date_transformer is None:
            with open(os.path.join(DATA_PATH, 'date_transformer.joblib'), 'rb') as file_:
                cls.date_transformer = joblib.load(file_)
        return cls.date_transformer
    
    @classmethod
    def load_minmax_scaler_total_purchases(cls):
        if cls.minmax_scaler_total_purchases is None:
            with open(os.path.join(DATA_PATH, 'minmax_scaler_total_purchases.joblib'), 'rb') as file_:
                cls.minmax_scaler_total_purchases = joblib.load(file_)
        return cls.minmax_scaler_total_purchases
    
    @classmethod
    def load_minmax_scaler_total_reviews(cls):
        if cls.minmax_scaler_total_reviews is None:
            with open(os.path.join(DATA_PATH, 'minmax_scaler_total_reviews.joblib'), 'rb') as file_:
                cls.minmax_scaler_total_reviews = joblib.load(file_)
        return cls.minmax_scaler_total_reviews
    
    @classmethod
    def load_minmax_scaler_purchases_last_60_days(cls):
        if cls.minmax_scaler_purchases_last_60_days is None:
            with open(os.path.join(DATA_PATH, 'minmax_scaler_purchases_last_60_days.joblib'), 'rb') as file_:
                cls.minmax_scaler_purchases_last_60_days = joblib.load(file_)
        return cls.minmax_scaler_purchases_last_60_days
    
    @classmethod
    def load_minmax_scaler_reviews_last_60_days(cls):
        if cls.minmax_scaler_reviews_last_60_days is None:
            with open(os.path.join(DATA_PATH, 'minmax_scaler_reviews_last_60_days.joblib'), 'rb') as file_:
                cls.minmax_scaler_reviews_last_60_days = joblib.load(file_)
        return cls.minmax_scaler_reviews_last_60_days
    
    @classmethod
    def load_age_group_encoder(cls):
        if cls.age_group_encoder is None:
            with open(os.path.join(DATA_PATH, 'age_group_encoder.joblib'), 'rb') as file_:
                cls.age_group_encoder = joblib.load(file_)
        return cls.age_group_encoder
    
    @classmethod
    def load_onehot_encoder(cls):
        if cls.onehot_encoder is None:
            with open(os.path.join(DATA_PATH, 'onehot_encoder.joblib'), 'rb') as file_:
                cls.onehot_encoder = joblib.load(file_)
        return cls.onehot_encoder
    
    @classmethod
    def encode_date(cls, feature):
        encoder = cls.load_date_transformer()
        return encoder.transform(feature)/cls.MAX_ACTIVE_DAYS
    
    @classmethod
    def encode_total_purchases(cls, feature):
        encoder = cls.load_minmax_scaler_total_purchases()
        return encoder.transform([[feature]])[0][0]
    
    @classmethod
    def encode_total_reviews(cls, feature):
        encoder = cls.load_minmax_scaler_total_reviews()
        return encoder.transform([[feature]])[0][0]
    
    @classmethod
    def encode_purchases_last_60_days(cls, feature):
        encoder = cls.load_minmax_scaler_purchases_last_60_days()
        return encoder.transform([[feature]])[0][0]
    
    @classmethod
    def encode_reviews_last_60_days(cls, feature):
        encoder = cls.load_minmax_scaler_reviews_last_60_days()
        return encoder.transform([[feature]])[0][0]
    
    @classmethod
    def encode_country(cls, feature):
        encoder = cls.load_onehot_encoder()
        return list(encoder.transform([[feature]]).toarray()[0])
    
    @classmethod
    def encode_age_group(cls, feature):
        encoder = cls.load_age_group_encoder()
        return encoder.transform([[feature]])[0][0]/cls.MAX_AGE_GROUP
    
    @classmethod
    def encode(cls, features):
        feature_vector = []
        active_since, total_purchases, total_reviews, purchases_last_60_days, reviews_last_60_days, country, age_group = features.split(',')
        feature_vector.append(cls.encode_date(active_since))
        feature_vector.append(cls.encode_total_purchases(total_purchases))
        feature_vector.append(cls.encode_total_reviews(total_reviews))
        feature_vector.append(cls.encode_purchases_last_60_days(purchases_last_60_days))
        feature_vector.append(cls.encode_reviews_last_60_days(reviews_last_60_days))
        feature_vector.append(cls.encode_country(country))
        feature_vector.append(cls.encode_age_group(age_group))
        return feature_vector
    
@app.route('/ping', methods=['GET'])
def ping():
    status = 200
    return Response(response={'HEALTH CHECK': 'OK'}, status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invoke():
    data = None

    # Transform payload in CSV to Pandas DataFrame.
    if request.content_type == 'text/csv':
        logger.info(f'RAW = {request.data}')
        data = request.data.decode('utf-8')
        logger.info(f'RAW  utf = {data}')
        logger.info(f'RAW  utf type = {type(data)}')
        data = StringIO(data)
        logger.info(f'RAW  stringio  = {data}')
        logger.info(f'RAW  stringio type = {type(data)}')
    else:
        return flask.Response(response='This Predictor only supports CSV data', status=415, mimetype='text/plain')

    logger.info('Invoked with {} records'.format(len(data)))
    
    encoded_data = FeatureEncoder.encode(data)
    encoded_data = np.hstack(encoded_data).tolist()
    logger.info(f'  encoded_data  = {encoded_data}')
    logger.info(f'  encoded_data type = {type(encoded_data)}')

    feature_vector = []

    return Response(response=feature_vector, status=200, mimetype='text/csv')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
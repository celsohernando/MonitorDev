import inspect
import logging
import datetime as dt
import math
from sqlalchemy.sql.sqltypes import TIMESTAMP,VARCHAR
import numpy as np
import pandas as pd
from io import StringIO

from iotfunctions.base import BaseTransformer
from iotfunctions import ui
from iotfunctions.ui import (UISingle, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti, UIMulti, UIExpression,
                 UIText, UIParameters)
from ibm_watson_machine_learning import APIClient

logger = logging.getLogger(__name__)

# Specify the URL to your package here.
# This URL must be accessible via pip install.
# Example assumes the repository is private.
# Replace XXXXXX with your personal access token.
# After @ you must specify a branch.

PACKAGE_URL = 'git+https://github.com/celsohernando/MonitorDev.git@starter_package'





class InvokeWMLCHF(BaseTransformer):
    '''
    Pass multivariate data in input_items to a regression function deployed to
    Watson Machine Learning. The results are passed back to the univariate
    output_items column.
    Credentials for the WML endpoint representing the deployed function are stored
    as pipeline constants, a name to lookup the WML credentials as JSON document.
    Example: 'my_deployed_endpoint_wml_credentials' referring to
    {
	    "apikey": "<my api key",
	    "url": "https://us-south.ml.cloud.ibm.com",
	    "space_id": "<my space id>",
	    "deployment_id": "<my deployment id">
    }
    This name is passed to InvokeWMLModel in wml_auth.
    '''
    def __init__(self, input_items, wml_auth, output_items):
        super().__init__()

        logger.debug(input_items)

        self.whoami = 'InvokeWMLModel'

        self.input_items = input_items
        self.set_entity_type('mfprinter1_li')

        if isinstance(output_items, str):
            self.output_items = [output_items]    # regression
        else:
            self.output_items = output_items      # classification

        self.wml_auth = wml_auth

        self.deployment_id = None
        self.apikey = None
        self.wml_endpoint = None
        self.space_id = None

        self.client = None

        self.logged_on = False


    def __str__(self):
        out = self.__class__.__name__
        try:
            out = out + 'Input: ' + str(self.input_items) + '\n'
            out = out + 'Output: ' + str(self.output_items) + '\n'

            if self.wml_auth is not None:
                out = out + 'WML auth: ' + str(self.wml_auth) + '\n'
            else:
                #out = out + 'APIKey: ' + str(self.apikey) + '\n'
                out = out + 'WML endpoint: ' + str(self.wml_endpoint) + '\n'
                out = out + 'WML space id: ' + str(self.space_id) + '\n'
                out = out + 'WML deployment id: ' + str(self.deployment_id) + '\n'
        except Exception:
            pass
        return out


    def login(self):

        # only do it once
        if self.logged_on:
            return

        # retrieve WML credentials as constant
        #    {"apikey": api_key, "url": 'https://' + location + '.ml.cloud.ibm.com'}
        c = None
        if isinstance(self.wml_auth, dict):
            wml_credentials = self.wml_auth
        elif self.wml_auth is not None:
            try:
                c = self._entity_type.get_attributes_dict()
            except Exception:
                c = None
            try:
                wml_credentials = c[self.wml_auth]
            except Exception as ae:
                raise RuntimeError("No WML credentials specified")
        else:
            wml_credentials = {'apikey': self.apikey , 'url': self.wml_endpoint, 'space_id': self.space_id}

        try:
            self.deployment_id = wml_credentials['deployment_id']
            self.space_id = wml_credentials['space_id']
            logger.info('Found credentials for WML')
        except Exception as ae:
            raise RuntimeError("No valid WML credentials specified")

        # get client and check credentials
        self.client = APIClient(wml_credentials)
        if self.client is None:
            #logger.error('WML API Key invalid')
            raise RuntimeError("WML API Key invalid")

        # set space
        self.client.set.default_space(wml_credentials['space_id'])

        # check deployment
        deployment_details = self.client.deployments.get_details(self.deployment_id, 1)
        # ToDo - test return and error msg
        logger.debug('Deployment Details check results in ' + str(deployment_details))

        self.logged_on = True


    def execute(self, df):

        logger.info('InvokeWML exec')
        logger.info('DF TYPES AT execute()')
        logger.info(df.dtypes)
        # Create missing columns before doing group-apply
        #df = df.copy().fillna('')
        df = df.copy().fillna(method='pad').fillna(0)
        #df = df.copy()
        logger.info('DF TYPES after df.copy().fillna('')')
        logger.info(df.dtypes)
        missing_cols = [x for x in (self.output_items) if x not in df.columns]
        for m in missing_cols:
            df[m] = None

        logger.info('DF TYPES after missing_cols')
        logger.info(df.dtypes)
        self.login()

        return super().execute(df)


    def _calc(self, df):

        inbuffer = StringIO()

        logger.info('INPUT DATAFRAME')
        logger.info(df.dtypes)
        logger.info('CAMBIO LOS FLOAT')
        df['duid']'= df['duid'].apply(lambda x: abs(int(x * 1000)))
        logger.info(df.head(10))
        df.to_csv(inbuffer, encoding='utf-8', index=True)
        logger.info(inbuffer.getvalue())

        if len(self.input_items) >= 1:

            index_nans = df[df[self.input_items].isna().any(axis=1)].index
            rows = df.loc[~df.index.isin(index_nans), self.input_items].values.tolist()
            INPUT_ITEMS = [x.upper() for x in self.input_items]
            logging.info(INPUT_ITEMS)
            scoring_payload = {
                'input_data': [{
                    'fields': INPUT_ITEMS,
                    'values': rows}]
            }
            logging.info(scoring_payload)
        else:
            logging.error("no input columns provided, forwarding all")
            return df

        results = self.client.deployments.score(self.deployment_id, scoring_payload)

        if results:
            # Regression
            if len(self.output_items) == 1:
                df.loc[~df.index.isin(index_nans), self.output_items] = \
                    np.array(results['predictions'][0]['values']).flatten()
            # Classification
            else:
                arr = np.array(results['predictions'][0]['values'])
                df.loc[~df.index.isin(index_nans), self.output_items[0]] = arr[:,0].astype(int)
                arr2 = np.array(arr[:,1].tolist())
                df.loc[~df.index.isin(index_nans), self.output_items[1]] = arr2.T[0]

        else:
            logging.error('error invoking external model')

        outbuffer = StringIO()
        df.to_csv(outbuffer, encoding='utf-8', index=True)
        logger.info('OUTPUT DATAFRAME')
        logger.info(df.dtypes)
        logger.info(outbuffer.getvalue())

        return df


    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(UIMultiItem(name = 'input_items', datatype=float,
                                  description = "Data items adjust", is_output_datatype_derived = True))
        inputs.append(UISingle(name='wml_auth', datatype=str,
                               description='Endpoint to WML service where model is hosted', tags=['TEXT'], required=True))

        # define arguments that behave as function outputs
        outputs=[]
        outputs.append(UISingle(name='output_items', datatype=float))
        return (inputs, outputs)

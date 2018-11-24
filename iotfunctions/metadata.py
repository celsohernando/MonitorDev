# *****************************************************************************
# Â© Copyright IBM Corp. 2018.  All Rights Reserved.
#
# This program and the accompanying materials
# are made available under the terms of the Apache V2.0
# which accompanies this distribution, and is available at
# http://www.apache.org/licenses/LICENSE-2.0
#
# *****************************************************************************

import logging
import datetime as dt
import json
import urllib3
import pandas as pd
from .db import Database, TimeSeriesTable
from sqlalchemy.sql.sqltypes import TIMESTAMP,VARCHAR
from ibm_db_sa.base import DOUBLE

logger = logging.getLogger(__name__)
    
class EntityType(object):
    '''
    Analytic service entity type    
    '''    
    
    log_table = 'KPI_LOGGING'
    checkpoint_table = 'KPI_CHECKPOINT'
    _timestamp = 'evt_timestamp'
    
    def __init__ (self,name,db, timestamp_col, *args, **kw):
        self.name = name
        if db is None:
            db = Database()
        self.db = db
        if not timestamp_col is None:
            self._timestamp = timestamp_col
        try:
            self.table = self.db.get_table(self.name)
        except KeyError:
            ts = TimeSeriesTable(self.name ,self.db, *args, **kw)
            self.table = ts.table
            self.table.create(self.db.connection)
        self.register()
        
    def get_params(self):
        
        params = {
                '_timestamp' : self._timestamp,
                'db' : self.db,
                'source_table' : self.table
                }
        return params
        
    def register(self):
        '''
        Register entity type so that it appears in the UI. Create a table for input data.
        
        Parameters
        ----------
        credentials: dict
            credentials for the ICS metadata service

        '''
        columns = []
        dates = []
        for c in self.db.get_column_names(self.table):
            if c not in ['logicalinterface_id','format','updated_utc']:
                data_type = self.table.c[c].type
                if isinstance(data_type,DOUBLE):
                    data_type = 'NUMBER'
                elif isinstance(data_type,TIMESTAMP):
                    data_type = 'TIMESTAMP'
                    dates.append(c)
                elif isinstance(data_type,VARCHAR):
                    data_type = 'LITERAL'
                else:
                    data_type = str(data_type)
                columns.append({ 
                        'name' : c,
                        'type' : 'METRIC',
                        'columnName' : c,
                        'columnType'  : data_type,
                        'tags' : None,
                        'transient' : False
                        })
        table = {}
        table['name'] = self.name
        table['dataItemDto'] = columns
        table['metricTableName'] = self.name
        table['metricTimestampColumn'] = self._timestamp
        table['schemaName'] = self.db.credentials['db2']['username']
        payload = [table]
        response = self.db.http_request(request='POST',
                                     object_type = 'entityType',
                                     object_name = self.name,
                                     payload = payload)

        msg = 'Metadata registerd for table %s '%self.name
        logger.debug(msg)
        return response
        
        
    def get_data(self,start_ts =None,end_ts=None,entities=None):
        '''
        Retrieve entity data
        '''
        (query,table) = self.db.query(self.name)
        if not start_ts is None:
            query = query.filter(table.c[self._timestamp] >= start_ts)
        if not end_ts is None:
            query = query.filter(table.c[self._timestamp] <= end_ts)  
        if not entities is None:
            query = query.filter(table.c.deviceid.in_(entities))
        df = pd.read_sql(query.statement, con = self.db.connection)
        
        return df       
        
        
    def get_log(self,rows = 100):
        
        query, log = self.db.query(self.log_table)
        query = query.filter(log.c.entity_type==self.name).\
                      order_by(log.c.timestamp_utc.desc()).\
                      limit(rows)
        df = self.db.get_query_data(query)
        return df
    
    
        
        
        
        
        

        
            
            
    
    
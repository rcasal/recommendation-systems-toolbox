# import libraries
from google.cloud import bigquery
import pandas as pd
import os
import datetime
from datetime import date, timedelta, datetime
from google.cloud import storage
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

# bring data from BQ
query = """
    DECLARE start_date DATE DEFAULT "2022-01-01";
    DECLARE end_date DATE DEFAULT "2022-01-31"; # CURRENT_DATE()

    CREATE OR REPLACE TABLE `mightyhive-data-science-poc.bads_labs_rs.cm_recommender_vea_raw`  as (
    WITH
    durations AS (
        --calculate pageview durations
        SELECT
            CONCAT( fullVisitorID,'-', 
                    CAST(visitNumber AS STRING),'-', 
                    CAST(hitNumber AS STRING) ) AS visitorId_session_hit,
            LEAD(time, 1) OVER (
                PARTITION BY CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING))
                ORDER BY time ASC ) - time AS pageview_duration
        FROM  `prd-spsa-data-ga.134299718.ga_sessions_*` , UNNEST(hits) AS hit 
        WHERE ( _TABLE_SUFFIX between REGEXP_REPLACE(CAST(start_date as STRING), "-", "") and 
            REGEXP_REPLACE(CAST(end_date as STRING), "-", "")) 
    ),

    full_table AS (
        SELECT
            fullVisitorID,
            visitorId,
            visitNumber,
            hitNumber,
            CONCAT(fullVisitorID,'-',CAST(visitNumber AS STRING)) AS visitor_visit_Id,
            productSKU as itemId,
            v2ProductName as v2ProductName,
            v2ProductCategory as v2ProductCategory,
            productVariant as productVariant,
            productBrand as productBrand,
            productPrice/1000000 as productPrice,
            IF(eCommerceAction.action_type = "1", IFNULL(dur.pageview_duration,1), 0) as product_click,
            IF(eCommerceAction.action_type = "2", IFNULL(dur.pageview_duration,1), 0) as detail_view,
            IF(eCommerceAction.action_type = "3",IFNULL(dur.pageview_duration,1),0) as add_to_cart,
            IF(eCommerceAction.action_type = "5",IFNULL(dur.pageview_duration,1),0) as check_out,
            IF(eCommerceAction.action_type = "6",IFNULL(dur.pageview_duration,1),0) as purchase_completed, 
            CASE
                WHEN eCommerceAction.action_type = "1" THEN 1
                WHEN eCommerceAction.action_type = "2" THEN 2
                WHEN eCommerceAction.action_type = "3" THEN 3
                WHEN eCommerceAction.action_type = "5" THEN 4
                WHEN eCommerceAction.action_type = "6" THEN 5
            ELSE 0
            END as rating 
        FROM `prd-spsa-data-ga.134299718.ga_sessions_*` t,
            UNNEST(hits) AS hit , UNNEST(product) as hits_product, durations as dur
        WHERE ( _TABLE_SUFFIX between REGEXP_REPLACE(CAST(start_date as STRING), "-", "") and 
            REGEXP_REPLACE(CAST(end_date as STRING), "-", ""))
    )

    SELECT   
        fullVisitorId as user_id,
        itemId as item_id,
        v2ProductName as item_info,
        rating as rating
    FROM full_table as ft
    JOIN durations dur ON 
    CONCAT(ft.fullVisitorID,'-', CAST(ft.visitNumber AS STRING),'-', CAST(ft.hitNumber AS STRING)) = dur.visitorId_session_hit
    )
"""

# vea 1 month
client = bigquery.Client()
df = client.query(query).to_dataframe()
df.to_csv("cm_vea_1month.csv", index=False)

#
R_df = df.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
R_df.head()


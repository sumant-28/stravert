import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import duckdb
import awswrangler as wr
import streamlit as st
import pydeck as pdk
from typing import Any
import boto3

def get_dfs(): 
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    # Get AWS credentials from Streamlit secrets
    try:
        aws_access_key = st.secrets["aws_access_key_id"]
        aws_secret_key = st.secrets["aws_secret_access_key"]
        aws_region = st.secrets.get("aws_region", "ap-southeast-2")
    except KeyError as e:
        st.error(f"AWS credentials not found in Streamlit secrets: {e}")
        st.info("Please add them to .streamlit/secrets.toml")
        st.stop()
    
    con.execute(f"""
    SET s3_region='{aws_region}';
    SET s3_access_key_id='{aws_access_key}';
    SET s3_secret_access_key='{aws_secret_key}';
    """)
    
    # Create boto3 session for awswrangler
    boto3_session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    source_prefix = "s3://sumant28-lastly/output/dt=*/*.parquet"
    json_files = wr.s3.list_objects(source_prefix, suffix=".parquet", boto3_session=boto3_session)
    
    a1 = []
    a2 = []
    a3 = []
    counter = len(json_files) - 1
    iterz = 0
    while (counter > 0) & (len(a1) < 1) | (len(a2) < 1) | (len(a3) < 1):
        q = f"""SELECT name FROM '{json_files[counter]}'"""
        test = con.execute(q).fetchdf()
        tx = test['name'].loc[0]
        if (tx == "Outdoor Cycling") & (len(a1) < 1):
            a1.append(json_files[counter])
        elif (tx == "Outdoor Walk") & (len(a2) < 1):
            a2.append(json_files[counter])
        elif (tx == "Outdoor Run") & (len(a3) < 1):
            a3.append(json_files[counter])
        counter -= 1
        iterz += 1
        
    d1 = con.execute(f"""SELECT * FROM '{a1[0]}'""").fetchdf()
    d2 = con.execute(f"""SELECT * FROM '{a2[0]}'""").fetchdf()
    d3 = con.execute(f"""SELECT * FROM '{a3[0]}'""").fetchdf()

    return d1, d2, d3

def ensure_data_loaded():
    """Ensure dataframes are loaded in session state - call this from every page"""
    if 'd1' not in st.session_state or 'd2' not in st.session_state or 'd3' not in st.session_state:
        with st.spinner("Loading data from S3..."):
            d1, d2, d3 = get_dfs()
            st.session_state.d1 = fix_all_timestamps_in_df(d1)
            st.session_state.d2 = fix_all_timestamps_in_df(d2)
            st.session_state.d3 = fix_all_timestamps_in_df(d3)
    return st.session_state.d1, st.session_state.d2, st.session_state.d3

def fix_timestamp(ts: Any) -> Any:
    local_tz = "Pacific/Auckland"
    """Convert a timestamp string or pd.Timestamp from UTC → local."""
    try:
        t = pd.to_datetime(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC").tz_convert(local_tz)
        else:
            t = t.tz_convert(local_tz)
        return t
    except Exception:
        return ts

def recursive_fix(obj):
    if isinstance(obj, list):
        return [recursive_fix(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: recursive_fix(v) for k, v in obj.items()}
    elif isinstance(obj, (str, pd.Timestamp, datetime)):
        return fix_timestamp(obj)
    else:
        return obj

def fix_all_timestamps_in_df(df: pd.DataFrame) -> pd.DataFrame:
    local_tz = "Pacific/Auckland"
    """Apply recursive timestamp fix to all columns in a DataFrame."""
    for col in df.columns:
        df[col] = df[col].apply(recursive_fix)
    return df

def explosion(string, df):
    s1 = df[string]
    s2 = s1.explode(string)
    output = pd.DataFrame(s2.tolist())
    return output

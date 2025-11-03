import streamlit as st
import utils
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import duckdb
import copy

st.set_page_config(page_icon="🏠", layout="wide")

# Display
st.title("🏠 Home")

# Check if dataframes are already in session state
if 'd1' not in st.session_state or 'd2' not in st.session_state or 'd3' not in st.session_state:
    d1, d2, d3 = utils.get_dfs()
    
    d1 = utils.fix_all_timestamps_in_df(d1)
    d2 = utils.fix_all_timestamps_in_df(d2)
    d3 = utils.fix_all_timestamps_in_df(d3)
    
    st.session_state.d1 = d1
    st.session_state.d2 = d2
    st.session_state.d3 = d3

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

d1, d2, d3 = utils.get_dfs()

d1 = utils.fix_all_timestamps_in_df(d1)
d2 = utils.fix_all_timestamps_in_df(d2)
d3 = utils.fix_all_timestamps_in_df(d3)

if d1 not in st.session_state:
    st.session_state.d1 = d1
if d2 not in st.session_state:
    st.session_state.d2 = d2
if d3 not in st.session_state:
    st.session_state.d3 = d3

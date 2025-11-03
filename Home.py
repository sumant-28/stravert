import streamlit as st
import utils

st.set_page_config(page_icon="🏠", layout="wide")

st.title("🏠 Home")

# Load data using the utility function
d1, d2, d3 = utils.ensure_data_loaded()

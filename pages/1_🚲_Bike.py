import streamlit as st
import plotly.express as px
import pandas as pd
import pydeck as pdk
from datetime import timedelta, datetime
from typing import Any
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import utils

local_tz = "Pacific/Auckland"

def explosion(string, df):
    s1 = df[string]
    s2 = s1.explode(string)
    output = pd.DataFrame(s2.tolist())
    return output

def fix_timestamp(ts: Any) -> Any:
    """Convert a timestamp string or pd.Timestamp from UTC → local."""
    try:
        t = pd.to_datetime(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC").tz_convert(local_tz)
        else:
            t = t.tz_convert(local_tz)
        return t
    except Exception:
        return ts  # leave non-timestamps unchanged

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
    """Apply recursive timestamp fix to all columns in a DataFrame."""
    for col in df.columns:
        df[col] = df[col].apply(recursive_fix)
    return df

def explosion(string, df):
    s1 = df[string]
    s2 = s1.explode(string)
    output = pd.DataFrame(s2.tolist())
    return output

# LOAD DATA FIRST - before set_page_config
d1, d2, d3 = utils.ensure_data_loaded()

local_tz = "Pacific/Auckland"

st.set_page_config(page_title="Bike", page_icon="🚲", layout="wide")

st.title("🚲 Bike")

# hrr = explosion('heartRateRecovery', d1)
# hrr = fix_all_timestamps_in_df(hrr)
r = explosion('route', d1)
r = fix_all_timestamps_in_df(r)
cd = explosion('cyclingDistance', d1)
cd = fix_all_timestamps_in_df(cd)
sc = explosion('stepCount', d1)
sc = fix_all_timestamps_in_df(sc)
ae = explosion('activeEnergy', d1)
ae = fix_all_timestamps_in_df(ae)
hrd = explosion('heartRateData', d1)
hrd = fix_all_timestamps_in_df(hrd)

#--- Define map ---
midpoint = (r["latitude"].mean(), r["longitude"].mean())

line_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": r[["longitude", "latitude"]].values.tolist()}],
    get_color=[255, 0, 0],
    width_scale=10,
    width_min_pixels=2,
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=r,
    get_position='[longitude, latitude]',
    get_color='[0, 0, 255, 160]',
    get_radius=20,
)

view_state = pdk.ViewState(
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=14,
    pitch=0,
)

deck = pdk.Deck(
    map_provider="carto",
    map_style="light",
    layers=[line_layer, point_layer],
    initial_view_state=view_state,
)

formatted = d1['start'].loc[0].strftime("%B %d, %Y at %-I:%M %p")
formatted = formatted[:-2] + formatted[-2:].lower()
st.subheader(f"{formatted}")
st.subheader(f"{round(d1['temperature'].loc[0]['qty'])} °C, {round(d1['humidity'].loc[0]['qty'])}% Humidity")

st.header("Route Map")
st.pydeck_chart(deck)

st.header("Metrics")

t1 = d1['end'][0] - d1['start'][0]
t2 = t1.to_pytimedelta()
t3 = f"{(t2.seconds//60)%60}m {t2.seconds % 60}s"
res = t2/d1['distance'][0]['qty']
res2 = timedelta(seconds=round(res.total_seconds()))
total_seconds = int(res2.total_seconds())
minutes = total_seconds // 60
seconds = total_seconds % 60
result2 = f"{(t2.seconds//60)%60}m {t2.seconds % 60}s"

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Distance", value=f"{round(d1['distance'][0]['qty'],2)} km")
    st.metric(label="Time", value=t3)
    st.metric(label="Calories", value=f"{round(d1['activeEnergyBurned'][0]['qty'])} Cal")


with col2:
    st.metric(label="Elevation Gain", value=f"{int(d1['elevationUp'][0]['qty'])} m")
    st.metric(label="Avg Speed", value=f"{round(d1['distance'][0]['qty']/t2.total_seconds()*3600, 2)} km/h")
    #st.metric(label="Avg Heart Rate", value=f"{round(sum(hrd['Avg'])/len(hrd))} bpm")

st.header("Average Heart Rate")

fig, ax = plt.subplots(figsize=(10, 5))

unique_to_cd = cd['date'][~cd['date'].isin(hrd['date'])]
unique_to_hrd = hrd['date'][~hrd['date'].isin(cd['date'])]
hrd2 = hrd.drop(unique_to_hrd.index)
cd2 = cd.drop(unique_to_cd.index)
s = hrd2['Avg']
cd_qty = list(cd2['qty'])
cd_qty.insert(0, 0)
reversed_s = s[::-1]
distance_km = np.cumsum(cd_qty)

# Plot line graph
ax.plot(distance_km[1:len(distance_km)], s, color='blue', marker='o', markersize=3, linewidth=1.5)

st.pyplot(fig)

st.metric(label="Avg Heart Rate", value=f"{int(sum(hrd['Avg'])/len(hrd))}")
st.metric(label="Max Heart Rate", value=f"{int(max(hrd['Max']))}")

st.header("Heart Rate Zones")

he = hrd['Avg']

bins = [0,127,158,174,189]

if len(bins) > 1 and len(he) > 0:
    categories = pd.cut(he, bins=bins)
    pie_data = categories.value_counts().reset_index()
    pie_data.columns = ["Range", "Count"]
    pie_data['Range'] = pie_data['Range'].astype(str).str.replace('[\(\)\[\]]','', regex=True)


    # Pie Chart
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta='Count:Q',
        color=alt.Color('Range:N', scale=alt.Scale(scheme='reds', reverse=True), sort='descending'),
        tooltip=['Range', 'Count']
    )
    st.altair_chart(pie_chart, use_container_width=True)

st.header("Speed")

cdate = cd2['date']
cdate2 = list(cdate)
cdate2.append(d1['end'].loc[0])
diffs = [(cdate2[i + 1] - cdate2[i]) for i in range(len(cdate2) - 1)]
diffs_seconds = [d.total_seconds() for d in diffs]
final = pd.DataFrame({'diffs_seconds': diffs_seconds, 'cd_qty': cd2['qty'], 'cdm': cd2['qty']*1000})
speed = (cd2['qty'] * 1000) / diffs_seconds
speed_kmh = speed * 3.6


fig, ax = plt.subplots(figsize=(10, 5))

# # Plot line graph
ax.plot(distance_km[1:len(distance_km)], speed_kmh, color='blue', marker='o', markersize=3, linewidth=1.5)

st.pyplot(fig)

st.subheader(f"Max Speed {round(max(speed_kmh),1)} km/h")

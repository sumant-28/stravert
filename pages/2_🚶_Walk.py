import streamlit as st
import plotly.express as px
import pandas as pd
import pydeck as pdk
from datetime import timedelta, datetime
from typing import Any
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import altair as alt
import utils

d1, d2, d3 = utils.ensure_data_loaded()

st.set_page_config(page_title="Walk", page_icon="🚶", layout="wide")

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

def pace_formatter(x, pos):
    minutes = int(x)
    seconds = int((x - minutes) * 60)
    return f"{minutes:02d}:{seconds:02d}"

def minsec_formatter(x, pos):
    minutes = int(x)
    seconds = int(round((x - minutes) * 60))
    return f"{minutes}:{seconds:02d}"

def calculate_km_splits_from_cumulative(cum_time, cum_dist):
    splits = []
    target_km = 1
    last_time = 0
    
    for i in range(1, len(cum_dist)):
        if cum_dist[i] >= target_km:
            # interpolate between this point and previous one
            dist_before = cum_dist[i - 1]
            dist_after = cum_dist[i]
            time_before = cum_time[i - 1]
            time_after = cum_time[i]
            
            fraction = (target_km - dist_before) / (dist_after - dist_before)
            split_time = time_before + fraction * (time_after - time_before)
            
            splits.append(split_time - last_time)  # time for this km split
            last_time = split_time
            target_km += 1
            
            if target_km > cum_dist[-1]:
                break
                
    return splits

def calculate_km_splits_from_cumulative2(cum_time, cum_dist):
    splits = []
    target_km = 1
    last_time = 0
    
    for i in range(1, len(cum_dist)):
        if cum_dist[i] >= target_km:
            # interpolate between this point and previous one
            dist_before = cum_dist[i - 1]
            dist_after = cum_dist[i]
            time_before = cum_time[i - 1]
            time_after = cum_time[i]
            
            fraction = (target_km - dist_before) / (dist_after - dist_before)
            split_time = time_before + fraction * (time_after - time_before)
            
            splits.append(split_time - last_time)  # time for this km split
            last_time = split_time
            #target_km += 1
            if target_km == np.floor(cum_dist[len(cum_dist)-1]):
                target_km = cum_dist[len(cum_dist)-1]
            else:
                target_km += 1
              
            if target_km > cum_dist[-1]:
                break
    if splits[len(splits)-1] < 0:
        splits = splits[0:len(splits)]
                
    return splits

def format_pace_minutes(pace_minutes):
    minutes = int(pace_minutes)
    seconds = int((pace_minutes - minutes) * 60)
    return f"{minutes:02d}:{seconds:02d}"

# Check data exists
# if 'sales_data' not in st.session_state:
#     st.error("⚠️ Please visit Home page first!")
#     st.stop()

st.title("🚶 Walk")

hrr = explosion('heartRateRecovery', d2)
hrr = fix_all_timestamps_in_df(hrr)
r = explosion('route', d2)
r = fix_all_timestamps_in_df(r)
cd = explosion('walkingAndRunningDistance', d2)
cd = fix_all_timestamps_in_df(cd)
sc = explosion('stepCount', d2)
sc = fix_all_timestamps_in_df(sc)
ae = explosion('activeEnergy', d2)
ae = fix_all_timestamps_in_df(ae)
hrd = explosion('heartRateData', d2)
hrd = fix_all_timestamps_in_df(hrd)

# --- Define map ---
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

formatted = d2['start'].loc[0].strftime("%B %d, %Y at %-I:%M %p")
formatted = formatted[:-2] + formatted[-2:].lower()
st.subheader(f"{formatted}")
st.subheader(f"{round(d2['temperature'].loc[0]['qty'])} °C, {round(d2['humidity'].loc[0]['qty'])}% Humidity")

st.header("Route Map")
st.pydeck_chart(deck)

st.header("Metrics")

t1 = d2['end'][0] - d2['start'][0]
t2 = t1.to_pytimedelta()
t3 = f"{(t2.seconds//60)%60}m {t2.seconds % 60}s"
res = t2/d2['distance'][0]['qty']
res2 = timedelta(seconds=round(res.total_seconds()))
total_seconds = int(res2.total_seconds())
minutes = total_seconds // 60
seconds = total_seconds % 60
result2 = f"{(t2.seconds//60)%60}m {t2.seconds % 60}s"

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Distance", value=f"{round(d2['distance'][0]['qty'],2)} km")
    st.metric(label="Time", value=t3)
    st.metric(label="Calories", value=f"{round(d2['activeEnergyBurned'][0]['qty'])} Cal")


with col2:
    st.metric(label="Steps", value=f"{format(round(sum(sc['qty'])), ",")}")
    st.metric(label="Elevation Gain", value=f"{int(d2['elevationUp'][0]['qty'])} m")
    st.metric(label="Avg Heart Rate", value=f"{round(sum(hrd['Avg'])/len(hrd))} bpm")

st.header("Workout Analysis")

cdate = list(cd['date'])
cdate.append(d2['end'].loc[0])
diffs = [cdate[i + 1] - cdate[i] for i in range(len(cdate)-1)]
sec = [int(td.total_seconds()) for td in diffs]
sec.insert(0, 0)
time_seconds = np.cumsum(sec)
cdqty = list(cd['qty'])
cdqty.insert(0, 0)
x1 = np.arange(0, len(cdate))
x2 = x1*60
distance_km = np.cumsum(cdqty)
time_diff = np.diff(time_seconds)
distance_diff = np.diff(distance_km)
distance_diff[distance_diff == 0]
pace_min_per_km = (time_diff / 60) / distance_diff
cumulative_distance_km = distance_km
cumulative_time_minutes = time_seconds/60
splits2 = calculate_km_splits_from_cumulative2(cumulative_time_minutes, cumulative_distance_km)
formatted_splits = [format_pace_minutes(p) for p in splits2]

pace_values = np.array(splits2)

x = distance_km[1:]  # skip first 0 km

    # Maximum pace for flipping bars
ymax = pace_values.max() + 0.5
ymin = pace_values.min() - 0.5

    # Bar heights flipped so faster pace is at top
bar_heights = ymax - pace_values

    # Function to format y-axis as min:sec
def minsec_formatter(x, pos):
    minutes = int(x)
    seconds = int(round((x - minutes) * 60))
    return f"{minutes}:{seconds:02d}"

values = bar_heights

x = np.arange(len(values))

full_width = 1.0
narrow_width = cumulative_distance_km[len(cumulative_distance_km) - 1] % 1

fig, ax = plt.subplots(figsize=(8, 4))

# Bars A–D (full width)
ax.bar(x[:-1], values[:-1], width=full_width, align='edge',
       color='lightblue', edgecolor='black')

# Final narrower bar (E)
ax.bar(x[-1], values[-1], width=narrow_width, align='edge',
       color='lightblue', edgecolor='black')

# Axis limits so first bar starts at 0, last bar ends exactly at its right edge
x_end = x[-1] + narrow_width
#ax.set_xlim(x[0], x_end)
ax.set_ylim(0, max(values) * 1.1)
ax.margins(0)

# Create ticks: integers plus one at the outer edge of the final bar
tick_positions = list(x) + [x_end]
tick_labels = [str(i) for i in x] + [f"{x_end:.1f}"]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

num_ticks = 6
locator = MaxNLocator(nbins=num_ticks)
ax.yaxis.set_major_locator(locator)

# Convert tick values back to original pace for labels
yticks = ax.get_yticks()
ytick_labels = [minsec_formatter(ymax - y, 0) for y in yticks]
ax.set_yticklabels(ytick_labels)

# Force tick locator to include the fractional tick explicitly
#ax.xaxis.set_major_locator(plt.FixedLocator(tick_positions))

# for i, bar in enumerate(bars):
#     pace = pace_values[i]
#     minutes = int(pace)
#     seconds = int(round((pace - minutes) * 60))
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{minutes}:{seconds:02d}",
#         ha='center', va='bottom')

# Labels and title
#ax.set_xlabel("X (integers + outer edge tick)")
ax.set_ylabel("/km")
#ax.set_title("Abutting Bars with Final Bar Fractionally Narrower and Edge-Aligned Tick")

plt.tight_layout()
st.pyplot(fig)

# st.subheader("Average Heart Rate")

# hrd['time_str'] = hrd['date'].dt.strftime('%H:%M:%S')

# # Create line figure
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=hrd['time_str'],
#     y=hrd['Avg'],
#     mode='lines+markers',
#     name='Pace'
# ))

# Update layout to avoid overlapping x-axis labels
# fig.update_layout(
#     xaxis=dict(
#         tickmode='array',      # Use a custom array for ticks
#         tickvals=hrd['time_str'][::5],   # Show every other tick (adjust as needed)
#         tickangle=45,          # Tilt labels for clarity
#     ),
#     yaxis_title='HR',
#     xaxis_title='Time',
#     margin=dict(l=40, r=40, t=40, b=80)
# )

# st.plotly_chart(fig, use_container_width=True)

st.header("Average Heart Rate Graph")

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

unique_to_cd = cd['date'][~cd['date'].isin(hrd['date'])]
unique_to_hrd = hrd['date'][~hrd['date'].isin(cd['date'])]
hrd2 = hrd.drop(unique_to_hrd.index)
s = hrd2['Avg']
cd_qty = list(cd['qty'])
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

st.header("Pace")

cdate = list(cd['date'])
cdate.append(d2['end'].loc[0])
diffs = [cdate[i + 1] - cdate[i] for i in range(len(cdate)-1)]
sec = [int(td.total_seconds()) for td in diffs]
sec.insert(0, 0)
time_seconds = np.cumsum(sec)
cdqty = list(cd['qty'])
cdqty.insert(0, 0)
x1 = np.arange(0, len(cdate))
x2 = x1*60
distance_km = np.cumsum(cdqty)
time_diff = np.diff(time_seconds)
distance_diff = np.diff(distance_km)
distance_diff[distance_diff == 0]
pace_min_per_km = (time_diff / 60) / distance_diff

cumulative_distance_km = distance_km
cumulative_time_minutes = time_seconds/60

splits = calculate_km_splits_from_cumulative(cumulative_time_minutes, cumulative_distance_km)
formatted_splits = [format_pace_minutes(p) for p in splits]

fig, ax = plt.subplots(figsize=(10, 5))

# Plot line graph
ax.plot(distance_km[1:len(distance_km)], pace_min_per_km, color='blue', marker='o', markersize=3, linewidth=1.5)

ax.set_ylim(8, 20)


# Invert y-axis so faster pace (smaller numbers) is at top
ax.invert_yaxis()

# Format y-axis as min:sec
ax.yaxis.set_major_formatter(mtick.FuncFormatter(minsec_formatter))

# Optional: nicely spaced y-axis ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# Labels and title
ax.set_ylabel("Pace (min/km)")
ax.set_title("Pace")

# Add grid for clarity
ax.grid(True, linestyle='--', alpha=0.5)

plt.fill_between(distance_km[1:len(distance_km)], pace_min_per_km, 20, color='blue', alpha=0.4)


# Show in Streamlit
st.pyplot(fig)
filtered = pace_min_per_km[(pace_min_per_km >= 0) & (pace_min_per_km <= 30)]
st.subheader(f"Avg Pace {minsec_formatter((sum(filtered)/len(filtered)),0)}  /km")


st.header("Cadence")

scm = sc.merge(cd, 'inner', 'date')
scm['cum'] = np.cumsum(scm['qty_y'])


# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# c = sc['qty']


ax.plot(scm['cum'], scm['qty_x'], color='blue', marker='o', markersize=3, linewidth=1.5)

plt.fill_between(scm['cum'], 0, scm['qty_x'], color='blue', alpha=0.4)


st.pyplot(fig)

averages = sc['qty']
#averages = averages[::-1]
d1 = d2['end'].loc[0] - d2['start'].loc[0]
total_time = d1.total_seconds()/60

n_full = int(total_time)
fractional = total_time - n_full

# Time-weighted sum
weighted_sum = np.sum(averages[:len(averages)])  # full minutes
if fractional > 0:
    weighted_sum += averages[len(averages)-1] * fractional  # add fractional weight

# Overall weighted average
overall_avg = weighted_sum / total_time

#st.metric(label="Avg Cadence", value=f"{int(max(df2['Max']))}")
st.metric(label="Max Cadence", value=f"{int(max(sc['qty']))}")
st.metric(label="Avg Cadence", value=f"{int(overall_avg)}")

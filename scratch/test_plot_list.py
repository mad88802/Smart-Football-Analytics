import plotly.express as px
import pandas as pd
import json

# Create a DataFrame
df = pd.DataFrame({
    'LD1': [1.0, 2.0, 3.0],
    'Target': ['A', 'B', 'A']
})

# Plot by passing lists directly
fig = px.scatter(
    x=df['LD1'].tolist(),
    y=df['LD1'].tolist(),
    color=df['Target'].tolist()
)

fig_json = json.loads(fig.to_json())
print("Type of x trace 0:", type(fig_json['data'][0]['x']))
print("Value of x trace 0:", fig_json['data'][0]['x'])

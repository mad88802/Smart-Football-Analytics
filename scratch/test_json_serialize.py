import plotly.express as px
import numpy as np
import json

# Create a dummy scatter plot with numpy arrays
fig = px.scatter(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))

# Try default to_json()
print("to_json output type of x:")
fig_json = json.loads(fig.to_json())
print(type(fig_json['data'][0]['x']), fig_json['data'][0]['x'])

# Try to_dict() + json.dumps()
print("\nto_dict() + json.dumps() output type of x:")
fig_dict = fig.to_dict()
fig_dict_json = json.loads(json.dumps(fig_dict))
print(type(fig_dict_json['data'][0]['x']), fig_dict_json['data'][0]['x'])

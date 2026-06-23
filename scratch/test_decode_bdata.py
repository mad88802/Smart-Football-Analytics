import base64
import numpy as np
import json
import plotly.express as px

def decode_plotly_json(obj):
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            # Decode binary base64 back to numpy array, then to list
            binary = base64.b64decode(obj['bdata'])
            arr = np.frombuffer(binary, dtype=obj['dtype'])
            return arr.tolist()
        else:
            return {k: decode_plotly_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_plotly_json(x) for x in obj]
    else:
        return obj

# Create plot
fig = px.scatter(x=np.array([1.0, 2.0, 3.0]), y=np.array([4.0, 5.0, 6.0]))

# Convert to dict
fig_dict = fig.to_dict()

# Clean dict
fig_dict_clean = decode_plotly_json(fig_dict)

# Serialize to JSON
clean_json = json.dumps(fig_dict_clean)
loaded = json.loads(clean_json)

print("Type of x after clean:", type(loaded['data'][0]['x']))
print("Value of x after clean:", loaded['data'][0]['x'])

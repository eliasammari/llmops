import plotly.express as px
df = pd.DataFrame({'humidity': [13.428571428571429, 50.375, 62.625, 72.21875, 100.0]})
fig = px.line(df, x=df.index, y="humidity")
fig.show()
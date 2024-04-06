import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np

model = pk.load(open("./model/model.pkl","rb"))
data = pd.read_csv("./x_data.csv")
st.title("Laptop Price Predictor")

company = st.selectbox(options=data["Company"].unique(),label="Brand")
type = st.selectbox(options=data["TypeName"].unique(),label="Type")
ram = st.selectbox(options=data["Ram"].unique(),label="Ram")
os = st.selectbox(options=data["os"].unique(),label="Oprating System")
weight = st.selectbox(options=data["Weight"].unique(),label="Weight")
touchScreen = st.selectbox(options=["Yes","No"],label="Touch Screen")
ipsPanel = st.selectbox(options=["Yes","No"],label="Ips Panel")
cpu = st.selectbox(options=data["Cpubrand"].unique(),label="Cpu")
gpu = st.selectbox(options=data["Gpu Brand"].unique(),label="Gpu")
ssd = st.selectbox(options=data["SSD"].unique(),label="SSD")
hdd = st.selectbox(options=data["HDD"].unique(),label="HDD")
ppi = st.selectbox(options=data["ppi"].unique(),label="Ppi")


if st.button("Predict"):
  try:
    if touchScreen == "Yes":
      touchScreen = 1
    else:
      touchScreen = 0
    if ipsPanel == "Yes":
      ipsPanel = 1 
    else:
      ipsPanel = 0
    query = np.array([[company,type,ram,weight,touchScreen,ipsPanel,ppi,cpu,ssd,hdd,gpu,os]],dtype=object)
    result = model.predict(query)
    result = np.exp(result)[0]
    result = f'{result:.2f}'
    st.success(result)
  except Exception as e:
    st.error("Internal Servor Error 🚨")





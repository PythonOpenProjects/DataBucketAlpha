# -*- coding: utf-8 -*-
"""
 -- Start STREAMLIT server --
 
cd my/path/to/streamlit/directoy/
streamlit run DataBucketAlpha.py
or
streamlit run DataBucketAlpha.py --server.port 80


 -- For HTTPS support: --
 
https://docs.streamlit.io/develop/concepts/configuration/https-support


 -- Create by hand the cerificate for SSL --
 
openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out cert.pem


 -- ./.streamlit/config.toml configuration fo SSL --
 
[server]
sslCertFile = 'cert.pem'
sslKeyFile = 'key.pem'


 -- For Dockerfile (and streamlit) requirements.txt -- 

streamlit
pandas
matplotlib
streamlit-folium
streamlit-extras
extra-streamlit-components
seaborn
sklearn
plotly
lime
numpy
pygwalker
streamlit_pandas_profiling
ydata_profiling
pivottablejs
dtale
mplcyberpunk
uuid
opencv-python
streamlit-extras
mitosheet
pydeck 
streamlit-aggrid
faker
lxml
fastparquet
st_aggrid (sttreamlit-aggrid>= 0.3.4.post3)
h5netcdf

 -- For Dockerfile -- 

FROM python:3.11.5

 ------ or try to use a previous lite version of Python ------ 

FROM python:3.8-slim

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["streamlit","run","DataBucketAlpha.py"]


 -- For Docker BUILD and RUN -- 

docker build -t databucketalpha:v1 .
docker images
docker run -p 8501:8501 databucketalpha:v1
"""


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium, folium_static
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pygwalker as pyw
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pivottablejs import pivot_ui
import dtale
import numpy
#https://github.com/dhaitz/mplcyberpunk
import mplcyberpunk
import ssl
import requests
import time
import os, sys
import shutil
import uuid
import datetime as dt
import cv2
from PIL import Image
import streamlit.components.v1 as components
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.jupyterlite import jupyterlite 
from datetime import datetime, timedelta
from mitosheet.streamlit.v1 import spreadsheet
from mitosheet.streamlit.v1.spreadsheet import _get_mito_backend
import pydeck as pdk
import inspect
from faker import providers
from faker import Faker
#import fastparquet

PROVIDER_MODULES = {"": ""}
PROVIDER_MODULES.update(
    {x[0]: x[1] for x in inspect.getmembers(providers, inspect.ismodule)}
)
PROVIDER_FUNCTIONS = {"": {"": ""}}
for k in PROVIDER_MODULES.keys():
    try:
        PROVIDER_FUNCTIONS.update(
            {
                k: {
                    x[0]: x[1]
                    for x in inspect.getmembers(
                        PROVIDER_MODULES[k].Provider, inspect.isfunction
                    )
                    if x[0] != "__init__"
                    and not x[0].startswith("_")
                    and "BaseProvider" not in str(x[1])
                }
            }
        )
    except AttributeError:
        PROVIDER_FUNCTIONS.update(
            {
                k: {
                    x[0]: x[1]
                    for x in inspect.getmembers(PROVIDER_MODULES[k], inspect.isfunction)
                    if x[0] != "__init__"
                    and not x[0].startswith("_")
                    and "BaseProvider" not in str(x[1])
                }
            }
        )
        

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "dark",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "button_face": "🌜"},

                    "dark":  {"theme.base": "light",
                              "button_face": "🌞"},
                    }


if ms.themes["current_theme"]=="dark":
    hide_streamlit_style = """
        <style>
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0.5rem;}
            .stSelectbox div[data-baseweb="select"] > div:first-child {
                    background-color: Chocolate;
                    border-color: #ff0000;
                }
            body
        </style>
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    
        """
else:
    hide_streamlit_style = """
        <style>
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0.5rem;}
            .stSelectbox div[data-baseweb="select"] > div:first-child {
                    background-color: gray;
                    border-color: #000000;
                }
            body
        </style>
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    
        """
        
st.set_page_config(
    page_title="Data Bucket Alpha",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

#START HIDE the TOP an burger menu!
st.markdown("""
<style>
	[data-testid="stDecoration"] {
		display: none;
	}
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}

</style>""",
unsafe_allow_html=True)

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True) 
#STOP HIDE the TOP an burger menu!

st.sidebar.title("Navigation Options 🧭")
# Display different functionalities
choice = st.sidebar.selectbox('Select....', options=["Home", "Load a Dataset", "Merge Datasets", "Load ODVs to MERGE in a Dataset", "Data Editor", "Dataframe Explorer", "Dataframe Analysis", "PLOT Service", "Map Service", "Model Prediction", "Aggrid Builder", "Pygwalker", "ydata_profiling", "pivottablejs", "Dtale", "MitoSheet", "Jupyterlite", "Fake Data Service", "Reset all"], index=0)


def mapservice():
    
    st.title(':blue[Map Service] 🗺️')
    if 'df' in st.session_state:
        df= st.session_state['df']
        cols_list = df.columns.tolist()
        optionLatituide = st.selectbox(
           "Choose Latitude column",
           (cols_list),
           index=None,
        )
        optionLongitude = st.selectbox(
           "Choose Longitude column",
           (cols_list),
           index=None,
        )
        optionName = st.selectbox(
           "Choose station name column",
           (cols_list),
           index=None,
        )
        onMapBasic = st.toggle('Map this!')
        onMap = st.toggle('Map this using Folium!')
        onMapPydeck = st.toggle('Map this using Pydeck!')
        
        if onMap and optionLatituide and optionLongitude:
            df_extr=df[[optionLatituide,optionLongitude,optionName]].copy()
            DF_Mysite_filtered=df_extr.drop_duplicates()
            DF_Mysite_filtered.columns = ['latitude', 'longitude','name']
            m = folium.Map(location=[DF_Mysite_filtered.latitude.mean(), DF_Mysite_filtered.longitude.mean()], zoom_start=7, control_scale=True)
            #Loop through each row in the dataframe
            for i,row in DF_Mysite_filtered.iterrows():
                #Setup the content of the popup
                iframe = folium.IFrame('Dataset name:' + str(row[2]), width=500, height=50)
                popup = folium.Popup(iframe, max_width=500)
                
                #Initialise the popup using the iframe
                popup = folium.Popup(iframe, min_width=300, max_width=300)
                
                #Add each row to the map
                folium.Marker(location=[row[0],row[1]],
                              popup = popup, c='Well Name').add_to(m)
            
            st_data = st_folium(m, width=800, height=500)
            
        if (onMapPydeck or onMapBasic) and optionLatituide and optionLongitude:
            if onMapPydeck:
                df_extr=df[[optionLongitude,optionLatituide]].copy()
                DF_Mysite_filtered=df_extr.drop_duplicates()
                DF_Mysite_filtered.columns = ['lon','lat']
                
                st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(
                        latitude=DF_Mysite_filtered.lat.mean(),
                        longitude=DF_Mysite_filtered.lon.mean(),
                        zoom=11,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                           'HexagonLayer',
                           data=DF_Mysite_filtered,
                           get_position='[lon, lat]',
                           radius=200,
                           elevation_scale=4,
                           elevation_range=[0, 1000],
                           pickable=True,
                           extruded=True,
                        ),
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=DF_Mysite_filtered,
                            get_position='[lon, lat]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                        ),
                    ],
                ))
            
            if onMapBasic:
                df_extr=df[[optionLatituide,optionLongitude]].copy()
                DF_Mysite_filtered=df_extr.drop_duplicates()
                DF_Mysite_filtered.columns = ['lat','lon']
                st.map(DF_Mysite_filtered)
            
    else:
        st.write('Please LOAD DATA')
    

def load_data_2_dataframe(file,separaval):
    '''
     Input Parameters
    ----------
    file : the file needs to be of type CSV
    Description
    -------
    Loads the data from the CSV file into a pandas dataframe
 
    Returns
    -------
    session_state containing dataframe df
    '''
    if 'df' not in st.session_state:
        if file.name.split(".")[-1] == "csv": 
            df = pd.read_csv(file,sep=separaval)
            for tmpcol in df.columns:
                cleanCol=function_cleaner(tmpcol)
                df.rename(columns={tmpcol: cleanCol}, inplace=True)

        elif file.name.split(".")[-1] == 'json':
            df = pd.read_json(file, lines=True)

        elif file.name.split(".")[-1] == 'xml':
            df = pd.read_xml(file)
             
        elif file.name.split(".")[-1] == 'parquet':
            df = pd.read_parquet(file, engine='fastparquet')
             
        elif file.name.split(".")[-1] == 'nc':
            import xarray as xr
            ds = xr.open_dataset(file)
            df = ds.to_dataframe()
            
        st.session_state['df'] = df
        st.dataframe(df)
        
        idrnd = uuid.uuid4()
        savename=str(time.strftime("%Y%m%d%H%M%S")+'-'+str(idrnd))
                
        csv, json, parquet, xml = st.columns(4)
        csv.download_button(
            label="Download data as CSV", data=df.to_csv(index=False), file_name="dw_data"+savename+".csv",
            mime="text/csv"
        )
        json.download_button(
            label="Download data as JSON", data=df.to_json(), file_name="dw_data"+savename+".json"
        )
        parquet.download_button(
            label="Download data as Parquet",
            data=df.to_parquet(index=False),
            file_name="dw_data"+savename+".parquet",
        )
 
    else:
        st.dataframe(st.session_state['df'])
        
    return st.session_state['df']

def resetall():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    st.write("The cache has been cleaned!")

def clean_data():
    if 'df' in st.session_state:
        df= st.session_state['df']
        count_rows_with_nan = df.isna().any(axis=1).sum()
        st.session_state['null_count']=count_rows_with_nan
        st.subheader("No. of rows containing null value",)
        # Display Null count
        st.metric("Null value", count_rows_with_nan, int(len(df)-count_rows_with_nan))    
        field = st.selectbox('Select Field to Analyze for Outliers', df.columns)
        # Plot and display in Streamlit
        if field:
            st.dataframe(detect_outliers(df, field))
def detect_outliers(df, field):
    Q1 = df[field].quantile(0.25)
    Q3 = df[field].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[field] < lower_bound) | (df[field] > upper_bound)]
def plot_data_with_outliers(df, field):
    outliers = detect_outliers(df, field)
    plt.style.use("cyberpunk")
    mplcyberpunk.add_glow_effects()
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
    fig, ax = plt.subplots()
    ax.boxplot(df[field])
    ax.scatter(outliers.index, outliers[field], color='red', label='Outliers')
    ax.set_title(f"Outliers in {field}")
    ax.legend()
    return fig           
def data_page():
    '''
     Display different graphs to understand the data
    '''
    if 'df' in st.session_state and st.session_state['df'] is  not None:
        #st.title('PLOT Service')
        st.title(':blue[PLOT Service] 📊')
        df= st.session_state['df']
        CorrGraph = st.toggle('Correlation graph')

        if CorrGraph:
            fig, ax = plt.subplots(figsize=(10,10))
            # Create the heatmap
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
            # Add a legend
            ax.legend()
            # Display the plot
            st.pyplot(fig)
                    
        #st.divider()  
        #st.write("🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺🔺")
        st.write("🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩")
        #st.write("🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻🔻")        
        selected_columns = st.multiselect('Select columns to plot:', df.columns)
        plotstyle = st.selectbox(
                                'Select PLOT STYLE (only for line charts)',
                                ("default",
                        		"classic",
                        		"Solarize_Light2",
                        		"bmh",
                        		"dark_background",
                        		"ggplot",
                        		"grayscale",
                        		"seaborn-v0_8",
                        		"seaborn-v0_8-bright",
                        		"seaborn-v0_8-pastel",
                                "cyberpunk"),
                                key="placeholder",
                                )

        plotLineChart = st.toggle('PLOT ALL PARAMS IN A SINGLE LINE CHART')
        ScatterGraph = st.toggle('PLOT ALL PARAMS IN a Scatter graph')
        AreaGraph = st.toggle('PLOT ALL PARAMS IN an Area graph')
        BarGraph = st.toggle('PLOT ALL PARAMS IN a Bar graph')
        singleplotLineChart = st.toggle('FOR EACH PARAM A SINGLE LINE CHART PLOT')
            
        if selected_columns is not None:
            # Create a figure
            if plotstyle=='':
                plt.style.use("cyberpunk")
                mplcyberpunk.add_glow_effects()
                mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
            else:
                plt.style.use(plotstyle)
            if plotLineChart:
                fig, ax = plt.subplots() 
                
                # Plot the data
                for column in selected_columns:
                    ax.plot(df[column], label=column)   
                    #st.write(df[column])
                # Set the axis labels
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Y Axis')    
                # Set the title
                ax.set_title('Line Chart')    
                # Add a legend
                ax.legend()    
                # Display the plot
                st.pyplot(fig)
            
            if ScatterGraph:
                dftemp = pd.DataFrame()
                dftemp=pd.concat([df[selected_columns]])
                st.scatter_chart(dftemp)
            if AreaGraph:
                dftemp = pd.DataFrame()
                dftemp=pd.concat([df[selected_columns]])
                st.area_chart(dftemp)
            if BarGraph:
                dftemp = pd.DataFrame()
                dftemp=pd.concat([df[selected_columns]])
                st.bar_chart(dftemp)
                
            if singleplotLineChart:
                  
                # Plot the data
                for column in selected_columns:
                    fig, ax = plt.subplots()  
                    ax.plot(df[column], label=column)    
                    # Set the axis labels
                    ax.set_xlabel('X Axis')
                    ax.set_ylabel('Y Axis')    
                    # Set the title
                    ax.set_title('Line Chart')    
                    # Add a legend
                    ax.legend()    
                    # Display the plot
                    st.pyplot(fig)
    else:
        st.write('Please LOAD DATA')
def reset_data():
    #if 'df' not in st.session_state:
    #    print('Dataframe not found')
 
    #else:
    #    del st.session_state['df']
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    st.write("The cache has been cleaned!")
    
    load_data()
    
def reset_dataOdv():
    if 'dfodv' not in st.session_state:
        print('Dataframe not found')
    else:
        del st.session_state['dfodv']
    
    load_Odvs()
    
    
def random_forest_feature_imp():
    if 'df' in st.session_state:
        df=st.session_state['df']
 
        X_train, y_train, X_test, y_test=create_test_train_data(df.columns[:-1])
 
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        plt.style.use("cyberpunk")
        mplcyberpunk.add_glow_effects()
        mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
 
        fig = plt.figure(figsize = (10, 5))     
        # creating the bar plot
        plt.bar(df.columns[:-1], model.feature_importances_, color ='maroon',
        width = 0.4)
        # Print the feature importances
        st.pyplot(fig)
    
def random_forest_model():
    # Train a random forest model
    if 'df' in st.session_state:
        df=st.session_state['df']
        model = RandomForestRegressor()
        selected_imp_columns_rf = st.multiselect('Select columns for training Random Forest:', df.columns[:-1],  key='rf')
        if selected_imp_columns_rf !=[]:
            X_train, y_train, X_test, y_test=create_test_train_data(selected_imp_columns_rf)
            model.fit(X_train, y_train)
            y_pred= model.predict(X_test)
            st.metric(" Mean Squared Error",np.round(mean_squared_error(y_test, y_pred),2))
            st.metric(" R Square",np.round(r2_score(y_test, y_pred),2))
            plt.style.use("cyberpunk")
            mplcyberpunk.add_glow_effects()
            mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
            fig1 = plt.figure(figsize = (10, 5))
            df_pred= pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            plt.plot(df_pred['actual'], label='Actual')
            plt.plot(df_pred['predictions'], label='predictions')
            plt.legend()
            st.pyplot(fig1)
def linear_regression_model():
    if 'df' in st.session_state:
        df=st.session_state['df']
        selected_imp_columns_lr = st.multiselect('Select columns for training Linear Regression:', df.columns[:-1], key='lr')
        if selected_imp_columns_lr !=[]:
            X_train, y_train, X_test, y_test=create_test_train_data(selected_imp_columns_lr) 
            model =LinearRegression()
            model.fit(X_train, y_train)
            y_pred=model.predict(X_test)
            st.metric(" Mean Squared Error",np.round(mean_squared_error(y_test, y_pred),2))
            st.metric(" R Square",np.round(r2_score(y_test, y_pred),2))
            plt.style.use("cyberpunk")
            mplcyberpunk.add_glow_effects()
            mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
            fig1 = plt.figure(figsize = (10, 5))
            df_pred= pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            plt.plot(df_pred['actual'], label='Actual')
            plt.plot(df_pred['predictions'], label='predictions')
            plt.legend()
            st.pyplot(fig1)
def create_test_train_data(selected_imp_columns):
    if 'df' in st.session_state:
        df= st.session_state['df']
        df_len=len(df)
        y_col=df.columns[-1]
        #st.write("in test train",selected_imp_columns)
        #df_col=selected_imp_columns
        train_len=int(.7*df_len)
        X_train=df.loc[:train_len, selected_imp_columns].copy()
        y_train=df.loc[:train_len,y_col].copy()
        X_test=df.loc[train_len:, selected_imp_columns].copy()
        y_test=df.loc[train_len:,y_col].copy()
        #st.write(X_test, y_test)
        return X_train, y_train, X_test, y_test
def lime_feature_imp():
    if 'df' in st.session_state:
        df= st.session_state['df']
        X_train, y_train, X_test, y_test=create_test_train_data(df.columns[:-1])
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        explainer=LimeTabularExplainer(training_data=np.array(X_train),
                              mode="regression",
                              feature_names=list(X_train.columns),
                              training_labels=np.array(y_train),
                              random_state=0)
        exp=explainer.explain_instance(X_test.iloc[0], model.predict, num_features=len(X_train.columns))
        st.pyplot(exp.as_pyplot_figure())
    
def model_selection(): 
    if 'df' in st.session_state:     
        #st.write(selected_imp_columns[0])
        st.title(':blue[Model Prediction]')
        #linear, RF, xgboost=st.tabs(['Linear Regression', 'Random Forest', 'XGBoost'])
        linear, RF=st.tabs(['Linear Regression', 'Random Forest'])    
        with linear:
            feature_imp, pred= st.columns(2)
            with feature_imp:
                st.subheader("Lime Feature Importances")
                lime_feature_imp()
            with pred:
                st.subheader("Prediction with Linear Regression")
                linear_regression_model()
        with RF:
            feature_imp, pred= st.columns(2)
            with feature_imp:
                st.subheader("Random Forest Feature Importance")
                random_forest_feature_imp()
            with pred:
                st.subheader("Prediction with Random Forest")
                random_forest_model()
    else:
        st.write('Please LOAD DATA')

def loadDtale():
    st.title(':blue[Dtale]')
    st.link_button("How to use", "https://www.youtube.com/watch?v=LRv6b1KDujI")
    if 'df' in st.session_state:
        df= st.session_state['df']
        d = dtale.show(df)
        d.open_browser()
        clean_data()
        st.write('A new web page has been opened for Dtale')
    else:
        st.write('Please LOAD DATA')
        
def loadpivottablejs():
    #st.write('PivoTableJS')
    st.title(':blue[PivoTableJS]')
    st.link_button("How to use", "https://pivottable.js.org/examples/index.html")
    if 'df' in st.session_state:
        df= st.session_state['df']
        t = pivot_ui(df)
        with open(t.src) as t:
            components.html(t.read(), width=1200, height=1000, scrolling=True)
        clean_data()
    else:
        st.write('Please LOAD DATA')

def loadydata_profiling():
    #st.write('Ydata Profiling')
    st.title(':blue[Ydata Profiling]')
    st.link_button("How to use", "https://docs.profiling.ydata.ai/latest/getting-started/concepts/")
    if 'df' in st.session_state:
        df= st.session_state['df']
        st.dataframe(df)
        pr = ProfileReport(df, title="Report")
        st_profile_report(pr)
        clean_data() 
    else:
        st.write('Please LOAD DATA')



def function_cleaner(x):
   x = x.replace(' ', '_')
   x = x.replace(':', '_')
   x = x.replace('[', '_')
   x = x.replace(']', '_')
   x = x.replace('.', '_')
   x = x.replace('/', '_')
   print('change '+str(x))
   return x



def loadDataEditor():
    if 'df' in st.session_state:
        df= st.session_state['df']
        
        #st.write('Data Editor')
        st.title(':blue[Data Editor] 📝')
        cols = st.columns(2)
        with cols[0]:
            selected_column = st.selectbox('Select column name to delete:', df.columns,index=None, placeholder="Select ...",)
            #st.write('')
            if st.button("Delete Column"):
                if selected_column is not None:
                    df.drop(columns=[selected_column], axis=1, inplace=True)
                
        with cols[1]:
            #st.write('Insert column name')
            
            title = st.text_input(
                "Insert column name to create a new one 👇",
                "",
                key="placeholder",
            )
            if st.button("Create Column"):
                if title != '':
                    df[title] = ''
                

            
        st.data_editor(df, num_rows="dynamic")
        #clean_data() 
    else:
        st.write('Please LOAD DATA')
        
def dataframexplorer():
    if 'df' in st.session_state:
        df= st.session_state['df']
        #st.write('Dataframe Explorer')
        st.title(':blue[Dataframe Explorer] 🕵️‍♂️')
        filtered_df = dataframe_explorer(df, case=False)
        st.dataframe(filtered_df, use_container_width=True) 
    else:
        st.write('Please LOAD DATA')
         
#--START MITO SHEET
def clear_mito_backend_cache():
    _get_mito_backend.clear()

# Function to cache the last execution time - so we can clear periodically
@st.cache_resource
def get_cached_time():
    # Initialize with a dictionary to store the last execution time
    return {"last_executed_time": None}

def try_clear_cache():
    # How often to clear the cache
    CLEAR_DELTA = timedelta(hours=12)
    current_time = datetime.now()
    cached_time = get_cached_time()
    # Check if the current time is different from the cached last execution time
    if cached_time["last_executed_time"] is None or cached_time["last_executed_time"] + CLEAR_DELTA < current_time:
        clear_mito_backend_cache()
        cached_time["last_executed_time"] = current_time

def mito():
    #st.write('MITO Sheet')
    st.title(':blue[MITO Sheet]')
    st.link_button("How to use", "https://docs.trymito.io/")
    if 'df' in st.session_state:
        df= st.session_state['df']
        new_dfs, code = spreadsheet(df)
        code = code if code else "# Edit the spreadsheet above to generate code"
        st.code(code)
        try_clear_cache()
 
    else:
        st.write('Please LOAD DATA')
#--END MITO SHEET

def jupyt():
    jupyterlite(600, 800)

#--START DATAFRAME ANALYSIS
# Gets additional value such as min / median / max etc.
def column_summary_plus(df):   
    result_df = []
    # Loop through each column in the DataFrame
    for column in df.columns:
        print(f"Start processing {column} col with {df[column].dtype} dtype")
        # Get column dtype
        col_dtype = df[column].dtype
        # Get distinct values and their counts
        value_counts = df[column].value_counts()
        distinct_values = value_counts.index.tolist()
        # Get number of distinct values
        num_distinct_values = len(distinct_values)
        # Get min and max values
        sorted_values = sorted(distinct_values)
        min_value = sorted_values[0] if sorted_values else None
        max_value = sorted_values[-1] if sorted_values else None
        # Get median value
        non_distinct_val_list = sorted(df[column].dropna().tolist())
        len_non_d_list = len(non_distinct_val_list)
        if len(non_distinct_val_list) == 0:
            median = None
        else:
            median = non_distinct_val_list[len_non_d_list//2]
        # Get average value if value is number
        if np.issubdtype(df[column].dtype, np.number):
            if len(non_distinct_val_list) > 0:
                average = sum(non_distinct_val_list)/len_non_d_list
                non_zero_val_list = [v for v in non_distinct_val_list if v > 0]
                average_non_zero = sum(non_zero_val_list)/len_non_d_list
            else:
                average = None
                average_non_zero = None
        else:
            average = None
            average_non_zero = None
        # Check if null values are present
        null_present = 1 if df[column].isnull().any() else 0
        # Get number of nulls and non-nulls
        num_nulls = df[column].isnull().sum()
        num_non_nulls = df[column].notnull().sum()
        # Distinct_values only take top 10 distinct values count
        top_10_d_v = value_counts.head(10).index.tolist()
        top_10_c = value_counts.head(10).tolist()
        top_10_d_v_dict = dict(zip(top_10_d_v,top_10_c))
        # Append the information to the result DataFrame
        result_df.append({'col_name': column, 
                          'col_dtype': col_dtype, 
                          'num_distinct_values': num_distinct_values, 
                          'min_value': min_value, 
                          'max_value': max_value,
                          'median_no_na': median, 
                          'average_no_na': average, 
                          'average_non_zero': average_non_zero,
                          'null_present': null_present, 
                          'nulls_num': num_nulls, 
                          'non_nulls_num': num_non_nulls,
                          'distinct_values': top_10_d_v_dict
        })
        
    result_df_out= pd.DataFrame(result_df)        
    return result_df_out

def dataframe_analysis():
    #st.write('MITO Sheet')
    st.title(':blue[Dataframe Analysis] 🎛️')
    if 'df' in st.session_state:
        df= st.session_state['df']
        #summary_df = column_summary(df)
        #st.write(summary_df)
        summary_df_plus = column_summary_plus(df)
        st.write(summary_df_plus)
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        # Perform univariate analysis on numerical columns
        for column in numerical_columns:
            # For continuous variables
            if len(df[column].unique()) > 10:  # Assuming if unique values > 10, consider it continuous
                fig1=plt.figure(figsize=(8, 6))
                sns.histplot(df[column], kde=True, color = "darkgreen", ec="black")
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')          
                st.pyplot(fig1)
            else:  # For discrete or ordinal variables
                fig1=plt.figure(figsize=(8, 6))
                ax = sns.countplot(x=column, data=df)
                plt.title(f'Count of {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                # Annotate each bar with its count
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 5), 
                                textcoords = 'offset points')
                st.pyplot(fig1)
    else:
        st.write('Please LOAD DATA')        
#--END DATAFRAME ANALYSIS       

def loadPgwalker():
    #st.write('Pgwalker')
    st.title(':blue[Pygwalker]')
    st.link_button("How to use", "https://www.youtube.com/watch?v=u0A-bcQHfmA")
    if 'df' in st.session_state:
        df= st.session_state['df']
        # Generate the HTML using Pygwalker
        pyg_html = pyw.to_html(df)
        components.html(pyg_html,width=900, height=1000, scrolling=True)
        clean_data() 
    else:
        st.write('Please LOAD DATA')        

def gob():
    st.title(':blue[Aggrid Builder]')
    if 'df' in st.session_state:
        df= st.session_state['df']
        from st_aggrid import AgGrid, GridOptionsBuilder
        grid_builder = GridOptionsBuilder.from_dataframe(df)
        grid_builder.configure_selection(selection_mode="multiple", use_checkbox=False)
        #grid_builder.configure_pagination(enabled=False, paginationAutoPageSize=False, paginationPageSize=3)
        grid_builder.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=10)
        grid_builder.configure_side_bar(filters_panel=True, columns_panel=False)
        grid_options = grid_builder.build()       
        AgGrid(data=df, gridOptions=grid_options, custom_css={
        "#gridToolBar": {
            "padding-bottom": "0px !important",
        }
    })
        #clean_data() 
    else:
        st.write('Please LOAD DATA') 
        
def fromfile(fn,dirname):
    counterHeader = 1
    try:
        f = open(dirname+'/'+fn,'r')
        while True:
            l=f.readline()
            if l.find('//')==-1:
                break
            counterHeader += 1
    except IOError:
        sys.exit(-1)

    data = pd.read_csv(dirname+'/'+fn,sep='\t',index_col=False, na_values=numpy.nan, skiprows = counterHeader-1) 
    data.columns = [c.replace(' ', '_') for c in data.columns]
    data['Cruise'].fillna(method='ffill', inplace = True)
    data['Station'].fillna(method='ffill', inplace = True)
    data['Type'].fillna(method='ffill', inplace = True)
    data['YYYY-MM-DDThh:mm:ss.sss'].fillna(method='ffill', inplace = True)
    data['Longitude_[degrees_east]'].fillna(method='ffill', inplace = True)
    data['Latitude_[degrees_north]'].fillna(method='ffill', inplace = True)
    data['LOCAL_CDI_ID'].fillna(method='ffill', inplace = True)
    data['EDMO_code'].fillna(method='ffill', inplace = True)
    data['Bot._Depth_[m]'].fillna(method='ffill', inplace = True)

    return data

def load_Odvs():
    '''
    loads the selected file into a dataframe
    stores the selected file and dataframe in st.session_state
    '''
    st.title(':blue[Merge ODVs in a dataset] 📒')
    st.write("You can load several ODVs to merge in a single dataframe")
    files = st.file_uploader("Upload ODV files", type=['txt'], accept_multiple_files=True)
    if 'dfodv' in st.session_state and st.session_state['dfodv'] is not None:
        st.write('Working...')      
    else:
        counterDummy=1
        idrnd = uuid.uuid4()
        dirname=str(time.strftime("%Y%m%d%H%M%S")+'-'+str(idrnd))
        os.mkdir(dirname)
        actualDir=os.getcwd()
        for uploaded_file in files:
            with open(os.path.join(dirname,uploaded_file.name),"wb") as f:
                f.write(uploaded_file.getbuffer())
        counterDummy=1
        li=[]
        for u in os.listdir(dirname):
            data=fromfile(u,dirname)
            li.append(data)
            counterDummy += 1
        if counterDummy >1:
            dfodv = pd.concat(li)         
            st.session_state['dfodv'] = dfodv
            st.dataframe(dfodv)
        shutil.rmtree(dirname)

#--START MERGER
def extractmultiple(file_to_extract,separaval):
    if file_to_extract.name.split(".")[-1] == "csv": 
        extracted_data = pd.read_csv(file_to_extract,sep=separaval)
    elif file_to_extract.name.split(".")[-1] == 'json':
         extracted_data = pd.read_json(file_to_extract, lines=True)
    elif file_to_extract.name.split(".")[-1] == 'xml':
         extracted_data = pd.read_xml(file_to_extract)
    elif file_to_extract.name.split(".")[-1] == 'parquet':
         extracted_data = pd.read_parquet(file_to_extract, engine='fastparquet')
    elif file_to_extract.name.split(".")[-1] == 'nc':
         import xarray as xr
         ds = xr.open_dataset(file_to_extract)
         extracted_data = ds.to_dataframe()      
    return extracted_data

def mergerdataset():
    '''
    loads the selected file into a dataframe
    stores the selected file and dataframe in st.session_state
    '''
    st.title(':blue[Merge Datasets (csv,json,xml,parquet,netcdf)] ♨️')
    st.caption("""
             Load and Merge (multiple upload)""")
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)         
    dataframes = []
    sepa = st.radio(
        "If CSV specify the separator",
        ["COMMA", "TAB", "COLON", "SEMICOLON"])        

    if uploaded_files:
        if sepa == 'COMMA':
            separaval=','
        if sepa == 'TAB':
            sepa='\t'
        if sepa == 'COLON':
            separaval=':'
        if sepa == 'SEMICOLON':
            separaval=';'
        
        for file in uploaded_files:
            file.seek(0)
            df = extractmultiple(file,separaval)
            dataframes.append(df)

        if len(dataframes) >= 1:
            merged_df = pd.concat(dataframes, ignore_index=True, join='outer')

        remove_duplicates = st.selectbox("Remove duplicate values ?", ["No", "Yes"])
        remove_nulls = st.selectbox("Remove null values in the dataset ?", ["No", "Yes"])
        if remove_duplicates == "Yes":
            merged_df.drop_duplicates(inplace=True)
        if remove_nulls == "Yes":
            merged_df.dropna(how="all", inplace=True)

        show_result = st.checkbox("Show Result", value=True)

        if show_result:
            st.write(merged_df)
            
        idrnd = uuid.uuid4()
        savename=str(time.strftime("%Y%m%d%H%M%S")+'-'+str(idrnd))
        csv, json, parquet, xml = st.columns(4)
        csv.download_button(
            label="Download cleaned data as CSV", data=merged_df.to_csv(index=False), file_name="cleaned_data"+savename+".csv",
            mime="text/csv"
        )
        json.download_button(
            label="Download cleaned data as JSON", data=merged_df.to_json(), file_name="cleaned_data"+savename+".json"
        )
        parquet.download_button(
            label="Download cleaned data as Parquet",
            data=merged_df.to_parquet(index=False),
            file_name="cleaned_data"+savename+".parquet",
        )
        
        show_resultProfile = st.checkbox("Show Profiling (this will take a while)", value=False)
        if show_resultProfile:
            pr = ProfileReport(merged_df, title="Report")
            st_profile_report(pr)
        
#--END MERGER

def load_data():
    '''
    loads the selected file into a dataframe
    stores the selected file and dataframe in st.session_state
    '''
    st.title(':blue[Load a dataset (csv,json,xml,parquet,netcdf)] ✨')
    # check if the dataframe df in st.session_state and is not blank
    if 'df' in st.session_state and st.session_state['df'] is not None:
        separaval=''
        df=load_data_2_dataframe(st.session_state['selected_file'],separaval)
        clean_data()
    #if the df does not exist in sesssion state then populate it       
    else:
        file = st.file_uploader("Upload a dataset", type=['csv','json','xml','parquet','nc'])
        sepa = st.radio(
            "If CSV specify the separator",
            ["COMMA", "TAB", "COLON", "SEMICOLON"])
        if file is not None:
            if sepa == 'COMMA':
                separaval=','
            if sepa == 'TAB':
                sepa='\t'
            if sepa == 'COLON':
                separaval=':'
            if sepa == 'SEMICOLON':
                separaval=';'
            st.session_state['selected_file'] = file
            df=load_data_2_dataframe(st.session_state['selected_file'],separaval)
            clean_data()           
        if 'null_count' in st.session_state:
            if st.session_state["null_count"] >0:
                null_action = st.radio(
                    'Select the action for handling Null Values',
                    ['Drop NA', 'Impute with 0', 'Impute with Mean', ])
                if null_action=='Drop NA':
                    #drop NA values in place
                    df= df.dropna(inplace=True)
                elif null_action=='Impute with 0':
                    # Fill missing values with a specific value
                    df = df.fillna(0, inplace=True)
                elif null_action=='Impute with 0':
                    # Fill missing values with mean of the column
                    df = df.fillna(df.mean(), inplace=True)
                    st.write(df)
                    
def add_row():
    element_id = uuid.uuid4()
    st.session_state["rows"].append(str(element_id))

def remove_row(row_id):
    st.session_state["rows"].remove(str(row_id))

def generate_row(row_id):
    row_container = st.empty()
    row_columns = row_container.columns(5)
    field_name = row_columns[0].text_input("Field Name", key=f"field_name-{row_id}")
    provider = row_columns[1].selectbox(
        "Provider", options=list(PROVIDER_MODULES.keys()), key=f"provider-{row_id}"
    )
    function = row_columns[2].selectbox(
        "Function",
        options=[""] + list(PROVIDER_FUNCTIONS.get(provider).keys()),
        key=f"function-{row_id}",
    )
    documentation = inspect.getdoc(PROVIDER_FUNCTIONS.get(provider).get(function))
    row_columns[3].text_area("Documentation", documentation, height=10, key=f"documentation-{row_id}")
    row_columns[4].button("🗑️", key=f"del_{row_id}", on_click=remove_row, args=[row_id])
    if field_name and provider and function:
        return {"field_name": field_name, "function": function}

fake = Faker()

def fakedata():
    st.title(':blue[Fake Data Service] 👺')
    if "rows" not in st.session_state:
        st.session_state["rows"] = []

    rows_collection = []
    
    for row in st.session_state["rows"]:
        row_data = generate_row(row)
        rows_collection.append(row_data)
    
    st.button("Add Row", on_click=add_row)
    
    if len(rows_collection) > 0:
        rows = st.slider("Number of rows", min_value=1, max_value=10000, value=1)  
        if st.button("Generate"):
            df = pd.DataFrame(
                {
                    row.get("field_name"): [
                        eval(f"fake.{row.get('function')}()") for i in range(rows)
                    ]
                    for row in rows_collection
                }
            )
    
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df)
    
            csv, json, parquet, xml = st.columns(4)
            csv.download_button(
                label="Download CSV", data=df.to_csv(index=False), file_name="data.csv"
            )
            json.download_button(
                label="Download JSON", data=df.to_json(), file_name="data.json"
            )
            parquet.download_button(
                label="Download Parquet",
                data=df.to_parquet(index=False),
                file_name="data.parquet",
            )
            try:
                import lxml
                xml.download_button(
                    label="Download XML", data=df.to_xml(index=False), file_name="data.xml"
                )
            except ModuleNotFoundError:
                print("If you would like to download your file as an XML document, please install lxml.")


def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"
        
def welcome():
    st.title('🪣 :blue[Data Bucket Alpha] 🪣')
    st.title(':blue[Welcome]') 
    btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
    st.write (':blue[\nChange Theme:]')
    st.button(btn_face, on_click=ChangeTheme)
    if ms.themes["refreshed"] == False:
      ms.themes["refreshed"] = True
      st.rerun()
      

if choice == "Load a Dataset":
    reset_data()
elif choice == "Load ODVs to MERGE in a Dataset":
    reset_dataOdv()
elif choice == "PLOT Service":
    data_page()
elif choice=="Model Prediction":
    model_selection()
elif choice=="Pgwalker":
    loadPgwalker()
elif choice=="ydata_profiling":
    loadydata_profiling()
elif choice=="pivottablejs":
    loadpivottablejs()
elif choice=="Dtale":
    loadDtale()
elif choice=="Data Editor":
    loadDataEditor()
elif choice=="Reset all":
    resetall()
elif choice=="Map Service":
    mapservice()
elif choice=="Dataframe Explorer":
     dataframexplorer()
elif choice=="Jupyterlite":
     jupyt()
elif choice=="MitoSheet":
     mito()
elif choice=="Dataframe Analysis":
     dataframe_analysis()
elif choice=="Merge Datasets":
     mergerdataset()
elif choice=="Fake Data Service":
     fakedata()
elif choice=="Aggrid Builder":
     gob()
elif choice=="Home":
    welcome()
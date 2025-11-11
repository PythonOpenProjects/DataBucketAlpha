# -*- coding: utf-8 -*-
"""
 -- Start STREAMLIT server --
 
cd Videos/tmpEr/Streamlit/PocketDataLakeHouse
streamlit run pdlh-lite.py



 -- PIP Install --
pip install streamlit
pip install time
pip install pandas
pip install uuid
pip install xarray
pip install pyspark
pip install pyarrow==11.0.0
pip install duckdb
pip install deltalake
pip install streamlit-extras


Tutorial Delta Lake
https://delta-io.github.io/delta-rs/usage/installation/
https://delta-io.github.io/delta-rs/usage/managing-tables/
https://delta-io.github.io/delta-rs/usage/querying-delta-tables/
https://delta-io.github.io/delta-rs/usage/writing/
https://delta-io.github.io/delta-rs/python/api_reference.html
https://delta-io.github.io/delta-rs/python/api_reference.html#deltalake.table.DeltaTable.files  (partitioning)
https://delta-io.github.io/delta-rs/usage/querying-delta-tables/
https://delta-io.github.io/delta-rs/python/usage.html


 -- For Dockerfile (and streamlit) requirements.txt -- 

streamlit
time
pandas
pyarrow
uuid
pyspark
xarray
duckdb
deltalake
streamlit-extras

 -- .streamlit/config.toml -- 

[server]
maxUploadSize = 2000
maxMessageSize = 2000




 -- For Dockerfile -- 

FROM python:3.11.5

 ------ or try to use a previous lite version of Python ------ 

FROM python:3.8-slim

WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["streamlit","run","pdlh-lite.py"]


 -- For Docker BUILD and RUN -- 

docker build -t pdlh-lite:v1 .
docker images
docker run -p 8501:8501 pdlh-lite:v1
"""


import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
from pathlib import Path
import xarray as xr
import time
import os
import shutil
import sys
import numpy
import duckdb
import seaborn as sns
import time
from deltalake import write_deltalake, DeltaTable
import matplotlib.pyplot as plt
import numpy as np
from streamlit_extras.dataframe_explorer import dataframe_explorer
import tempfile
 
importname=''

st.set_page_config(
    page_title="Pocket Data Lakehouse",
    page_icon="🗃️",
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
                    background-color: #FADA7A;
                    border-color: #780C28;
                }
            body
        </style>
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    
        """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True) 
#STOP HIDE the TOP an burger menu!

#https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.sidebar.title("MENU 🗃️")

choice = st.sidebar.selectbox('Select....', options=["🗃️ Home","🧊 📝 Import Data Delta Lake","🧊 📒 Access Data Delta Lake","🧊 📒 Explore Data Delta Lake","🧊 🛠️ Partition Data Delta Lake", "🧊 🦆 SQL on Delta Lake using DuckDB", "🧊 📈 Instruments Delta Lake", "🧊 🗑️ Delete Data on Delta Lake", "📒 Load ODVs to MERGE in a Dataset"], index=0)

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"
        
def welcome():
    st.title(':blue[Pocket Data Lakehouse] 🗃️')
    st.title(':blue[Welcome]')
    
    btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
    st.write (':blue[\nChange Theme:]')
    st.button(btn_face, on_click=ChangeTheme)
    
    if ms.themes["refreshed"] == False:
      ms.themes["refreshed"] = True
      st.rerun()
      
      

def load_data_2_dataframe(file,separaval):

    try:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "csv":
            df = pd.read_csv(file, sep=separaval)
        elif file_extension == "json":
            df = pd.read_json(file, lines=True)
        elif file_extension == "xml":
            df = pd.read_xml(file)
        elif file_extension == "parquet":
            df = pd.read_parquet(file, engine='fastparquet')
        elif file_extension == "nc":
            ds = xr.open_dataset(file)
            df = ds.to_dataframe()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            st.toast('Unsupported file format!', icon='🗑️')
        return df

    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def parquet_append(filepath, df: pd.DataFrame) -> None:
    
    try:
        table_original_file = pq.read_table(source=filepath, pre_buffer=False, use_threads=True, memory_map=True)
        table_to_append = pa.Table.from_pandas(df)
        
        if table_original_file.schema != table_to_append.schema:
            table_to_append = table_to_append.cast(table_original_file.schema)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_filepath = temp_file.name
        
        with pq.ParquetWriter(temp_filepath, table_original_file.schema, compression='snappy') as handle:
            handle.write_table(table_original_file)
            handle.write_table(table_to_append)
        
        os.replace(temp_filepath, filepath)
    
    except Exception as e:
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise


def importpageDlake():
    st.title(':blue[Delta Lake] 🧊')
    st.title(':blue[Load a dataset (csv,json,xml,parquet,netcdf)] 📝')
    file = st.file_uploader("Upload a dataset", type=['csv','json','xml','parquet','nc'])
    idrnd = uuid.uuid4()
    execute=0
    import os
    files = [ f.path for f in os.scandir('deltalake') if f.is_dir() ]
    cols = st.columns(4)
    with cols[1]:
        loadData = st.selectbox(
           "Append to an existing dataset",
           (files),
           index=None,
        )
    with cols[0]:
        placeholder = st.empty()
        dataname = placeholder.text_input("Write a label for a new dataset",importname)
        separa = st.radio(
            "If CSV, specify the separator",
            ["TAB", "COMMA", "COLON", "SEMICOLON"])
        
        if separa == 'TAB':
            separaval='\t'
        if separa == 'COMMA':
            separaval=','
        if separa == 'COLON':
            separaval=':'
        if separa == 'SEMICOLON':
            separaval=';'

    with cols[2]:   
        st.write('New data with a different structure? (merge data in a new dataset)')
        differentdata = st.toggle('YES')
        
    with cols[3]:
        if st.button("EXECUTE"):
            execute=1
        
    if loadData:
        tmploadData=loadData.split("\\")
        
        if differentdata:
            newdataname=tmploadData[1]+'_'+str(time.strftime("%Y%m%d%H%M%S"))
        else:
            newdataname=tmploadData[1]
            
        dataname = placeholder.text_input('text', value=newdataname, key=1)
    if file is not None and dataname !='' and execute==1:
        df=load_data_2_dataframe(file,separaval)
        
        #check if the dataset already exist
        import os.path
        path = 'deltalake/'+dataname        
        check_file = os.path.isdir(path)
             
        if loadData:
            if differentdata:           
                dtOld=DeltaTable('deltalake/'+tmploadData[1])
                dataVersions=dtOld.version()
                versionlist=[]
                x = range(1, dataVersions+1)
                for n in x:
                    lastVersion=n
                
                dtOld.load_as_version(lastVersion)
                oldToMerge=dtOld.to_pandas()
                df_new = pd.concat([oldToMerge, df])             
                dt_new=str('deltalake/'+dataname)
                write_deltalake(dt_new, df_new)
                dt_new=DeltaTable('deltalake/'+dataname)
                write_deltalake(dt_new, df_new, mode="overwrite")               
            else:
                dt=DeltaTable('deltalake/'+dataname)
                df.columns = df.columns.str.lower()
                write_deltalake(dt, df, mode="append")
        else:
            dt=str('deltalake/'+dataname)
            df.columns = df.columns.str.lower()
            write_deltalake(dt, df)
            dt=DeltaTable('deltalake/'+dataname)
            write_deltalake(dt, df, mode="overwrite")
     
        execute=0
        
        
def reset_dataOdv():
    if 'dfodv' not in st.session_state:
        print('Dataframe not found')
 
    else:
        del st.session_state['dfodv']
    
    load_Odvs()
    
def load_Odvs():
    '''
    loads the selected file into a dataframe
    stores the selected file and dataframe in st.session_state
    '''
    st.title(':blue[Merge ODVs in a dataset] 📒')
    st.write("You can load several ODVs to merge in a single dataframe")
    # check if the dataframe df in st.session_state and is not blank
    files = st.file_uploader("Upload ODV files", type=['txt'], accept_multiple_files=True)
    if 'dfodv' in st.session_state and st.session_state['dfodv'] is not None:
        st.write('Working...')
    #if the df does not exist in sesssion state then populate it       
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

def fromfile(fn,dirname):
    """load 1 ODV file entirely into Odv object: dataframe + metadata fields for columns"""

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

    data = pd.read_csv(dirname+'/'+fn,sep='\t',index_col=False, na_values=numpy.nan, skiprows = counterHeader-1) #, parse_dates=[3], infer_datetime_format=True, date_parser=odvdatetime)
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
               

def sqlpageduckDeltaLake():
    st.title(':blue[SQL in Delta Lake 🧊 using Duck DB] 🦆')   
    directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
    execute=0
    cols = st.columns(3)
    with cols[0]:
        loadDataDuck = st.selectbox(
           "Choose dataset",
           (directories),
           index=None,
        )
        
    if loadDataDuck:
        dt=DeltaTable("deltalake/"+loadDataDuck)
        dataFiles=dt.files()
        with cols[1]:
            loadDataFiles = st.selectbox(
               "Choose file",
               (dataFiles),
               index=None,
            )
        
        if loadDataFiles:           
            duckdb.read_parquet('deltalake/'+loadDataDuck+'/'+loadDataFiles)
            tmpViewSplittedDuck=str(loadDataDuck)
            txtQueryDuck = st.text_area(
            "Write here your query",
            "SELECT * FROM 'deltalake/"+str(tmpViewSplittedDuck)+"/"+loadDataFiles+"' LIMIT 5",height=300
            )
            st.write('')         
            # Esecuzione della query
            if st.button("EXECUTE QUERY"):
                with st.spinner("The query is running..."):
                    df2Duck = duckdb.sql(txtQueryDuck).df()
                if df2Duck is not None:
                    st.success("Query executed!")
                    st.dataframe(df2Duck) 


def datareadpageDeltaLake():
    """
    Streamlit interface to interact with data stored in Delta Lake.
    Allows you to select a dataset, view versions, clone, download, and delete rows.
    """
    st.title(':blue[Delta Lake] 🧊 Access Data 📒')
    # Carica le directory disponibili
    try:
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
    except Exception as e:
        st.error("Error while loading directories.")
        return
    # Interfaccia utente
    cols = st.columns(4)
    with cols[0]:
        loadData = st.selectbox(
            "Choose dataset",
            directories,
            index=None,
        )
    with cols[1]:
        st.write('')
        st.write(':blue[Do you want to download the data?]')
        downloaddata = st.toggle('Download data CSV format', key="1")
    with cols[2]:
        st.write('')
        st.write(':blue[Do you want to see the data?]')
        tableloaddata = st.toggle('Show table data', key="2")
    with cols[3]:
        st.write('')
        st.write(':red[Clone this version as a new dataset?]')
        placeholder = st.empty()
        dataname = placeholder.text_input("Write a label for the cloned dataset", '')
        clonedata = st.toggle('CLONE IT!', key="3")

    if loadData:
        try:
            dt = DeltaTable("deltalake/" + loadData)
            dataVersions = dt.version()
            versionlist = list(range(1, dataVersions + 1))

            loadDataVersion = st.selectbox(
                "Choose version",
                versionlist,
                index=None,
            )
            if loadDataVersion:
                dt.load_as_version(int(loadDataVersion))
                output = dt.to_pandas()
                # Clonazione del dataset
                if clonedata:
                    if not dataname:
                        st.warning('Please, write a label for the new dataset')
                    else:
                        try:
                            clonedt = f'deltalake/{dataname}'
                            cloned = dt.to_pandas()
                            cloned.columns = cloned.columns.str.lower()
                            write_deltalake(clonedt, cloned, mode="overwrite")
                            st.success(f'Cloned: a new dataset has been created using the label: {dataname}')
                        except Exception as e:
                            st.error("Error while cloning the dataset.")

                # Download dei dati
                if downloaddata:
                    try:
                        idrnd = uuid.uuid4()
                        savename = f"{time.strftime('%Y%m%d%H%M%S')}-{idrnd}"
                        csv, dummy1, dummy2 = st.columns(3)
                        csv.download_button(
                            label="Download Saved data as CSV",
                            data=output.to_csv(index=False),
                            file_name=f"Saved_data_{savename}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error("Error while data downloading.")

                # Visualizzazione dei dati
                if tableloaddata:
                    try:
                        st.write(dataframe_explorer(output, case=False))
                        st.divider()
                        st.subheader(':red[Delete data by conditions]')
                        cols = st.columns(6)
                        with cols[0]:
                            columnlist = st.selectbox(
                                "Choose column",
                                output.columns,
                                index=None,
                            )

                        with cols[1]:
                            if columnlist:
                                conditionslist = st.selectbox(
                                    "Choose condition",
                                    ('', '>', '<', '==', '!=', 'between'),
                                    index=None,
                                )
                                with cols[2]:
                                    if conditionslist:
                                        paramA = st.text_input('', key="10")
                                        if conditionslist == 'between':
                                            st.write(' AND ')
                                            paramB = st.text_input('', key="20")

                        # Eliminazione delle righe
                        with cols[0]:
                            if conditionslist:
                                if st.button('Delete rows'):
                                    try:
                                        myquery = f"{columnlist} {conditionslist} {paramA}"
                                        if conditionslist == 'between':
                                            myquery = f"{columnlist} >= {paramA} and {columnlist} <= {paramB}"

                                        dt.delete(myquery)
                                        st.toast('Please, reload the new version of the dataset', icon='👍')
                                        st.toast(f'Rows have been deleted using the query: {myquery}', icon='👍')
                                    except Exception as e:
                                        st.error("Error while deleting rows.")
                    except Exception as e:
                        st.error("Error while displaying data.")

        except Exception as e:
            st.error("Error while loading dataset.")
                            


def dataexplorepageDeltaLake():
    """
    Streamlit interface to interact with data stored in Delta Lake.
    Allows you to select a dataset, view versions, clone, download, and delete rows.
    """
    st.title(':blue[Delta Lake] 🧊 Explore Data 📒')
    # Carica le directory disponibili
    try:
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
    except Exception as e:
        st.error("Error while loading directories.")
        return
    # Interfaccia utente
    cols = st.columns(3)
    with cols[0]:
        loadData = st.selectbox(
            "Choose dataset",
            directories,
            index=None,
        )
    with cols[1]:
        st.write('')
        st.write(':blue[Do you want to download the data?]')
        downloaddata = st.toggle('Download data CSV format', key="1")
    with cols[2]:
        st.write('')
        st.write(':blue[Do you want to see the data?]')
        tableloaddata = st.toggle('Show table data', key="2")
    

    if loadData:
        try:
            dt = DeltaTable("deltalake/" + loadData)
            dataVersions = dt.version()
            versionlist = list(range(1, dataVersions + 1))

            loadDataVersion = st.selectbox(
                "Choose version",
                versionlist,
                index=None,
            )
            if loadDataVersion:
                dt.load_as_version(int(loadDataVersion))
                output = dt.to_pandas()
                

                # Download dei dati
                if downloaddata:
                    try:
                        idrnd = uuid.uuid4()
                        savename = f"{time.strftime('%Y%m%d%H%M%S')}-{idrnd}"
                        csv, dummy1, dummy2 = st.columns(3)
                        csv.download_button(
                            label="Download Saved data as CSV",
                            data=output.to_csv(index=False),
                            file_name=f"Saved_data_{savename}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error("Error while data downloading.")

                # Visualizzazione dei dati
                if tableloaddata:
                    try:
                        st.write(dataframe_explorer(output, case=False))
                        st.divider()
                        
                    except Exception as e:
                        st.error("Error while displaying data.")
                        
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
                selected_columns = st.multiselect('Select columns to plot:', output.columns)
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
                    plt.style.use(plotstyle)
                    if plotLineChart:
                        fig, ax = plt.subplots() 
                        
                        # Plot the data
                        for column in selected_columns:
                            ax.plot(output[column], label=column)   
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
                        dftemp=pd.concat([output[selected_columns]])
                        st.scatter_chart(dftemp)
                    if AreaGraph:
                        dftemp = pd.DataFrame()
                        dftemp=pd.concat([output[selected_columns]])
                        st.area_chart(dftemp)
                    if BarGraph:
                        dftemp = pd.DataFrame()
                        dftemp=pd.concat([output[selected_columns]])
                        st.bar_chart(dftemp)
                        
                    if singleplotLineChart:
                          
                        # Plot the data
                        for column in selected_columns:
                            fig, ax = plt.subplots()  
                            ax.plot(output[column], label=column)    
                            # Set the axis labels
                            ax.set_xlabel('X Axis')
                            ax.set_ylabel('Y Axis')    
                            # Set the title
                            ax.set_title('Line Chart')    
                            # Add a legend
                            ax.legend()    
                            # Display the plot
                            st.pyplot(fig)

        except Exception as e:
            st.error("Error while loading dataset.")




                    
def dataPartitionDeltaLake():
    st.title(':blue[Delta Lake] 🧊 Partition Data 🛠️')

    # Carica le directory disponibili
    try:
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
    except Exception as e:
        st.error("Error while loading directories.")
        return

    # Interfaccia utente
    cols = st.columns(4)
    with cols[0]:
        loadData = st.selectbox(
            "Choose dataset",
            directories,
            index=None,
        )

    if loadData:
        try:
            dt = DeltaTable("deltalake/" + loadData)
            dataVersions = dt.version()
            versionlist = list(range(1, dataVersions + 1))

            loadDataVersion = st.selectbox(
                "Choose version",
                versionlist,
                index=None,
            )
            if loadDataVersion:
                dt.load_as_version(int(loadDataVersion))
                output = dt.to_pandas()
                column_names = output.columns

                columsPartition = st.selectbox(
                    "Choose column for partitioning",
                    column_names,
                    index=None,
                )
                if columsPartition:
                    st.write('')
                    st.write(':blue[Do you want to partition the data?]')
                    partitiondata = st.toggle('YES, make the partition!', key="1")

                    if partitiondata:
                        try:
                            # Creazione di una cartella temporanea
                            temp_dir = f'tmp/{uuid.uuid4()}'
                            os.makedirs(temp_dir, exist_ok=True)

                            # Partizionamento e scrittura dei dati
                            write_deltalake(temp_dir, output, partition_by=[columsPartition])

                            # Compressione dei dati partizionati
                            zip_filename = f'{temp_dir}.zip'
                            shutil.make_archive(temp_dir, 'zip', temp_dir)

                            # Download del file ZIP
                            with open(zip_filename, "rb") as fp:
                                btn = st.download_button(
                                    label=f"Download partitioned files as ZIP (using the field: {columsPartition})",
                                    data=fp,
                                    file_name=os.path.basename(zip_filename),
                                    mime="application/zip"
                                )

                            # Pulizia dei file temporanei
                            shutil.rmtree(temp_dir)
                            os.remove(zip_filename)

                        except Exception as e:
                            st.error("Error while partitioning data.")
                            # Pulizia dei file temporanei in caso di errore
                            if os.path.exists(temp_dir):
                                shutil.rmtree(temp_dir)
                            if os.path.exists(zip_filename):
                                os.remove(zip_filename)

        except Exception as e:
            st.error("Error while loading dataset.")    


def DeltaLakedeletepage():
    """
    Streamlit interface to delete datasets stored in Delta Lake.
    Allows you to select a dataset and double-confirm the deletion.
    """
    st.title(':blue[Delete Data from Delta Lake] 🧊 🗑️')

    # Carica le directory disponibili
    try:
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
    except Exception as e:
        st.error("Error while loading directories.")
        return

    # Interfaccia utente
    cols = st.columns(3)
    with cols[0]:
        delData = st.selectbox(
            "Choose dataset",
            directories,
            index=None,
        )

    if delData:
        st.write(':blue[Do you want to delete this dataset?]')
        deletedataconfirmA = st.toggle('Yes', key="1")

        if deletedataconfirmA:
            st.write(':blue[Are you sure? You want to delete this file?]')
            deletedataconfirmB = st.toggle('Yes, let\'s go!!!!', key="2")

            if deletedataconfirmB:
                try:
                    # Eliminazione del dataset
                    dataset_path = os.path.join("deltalake", delData)
                    if os.path.exists(dataset_path):
                        shutil.rmtree(dataset_path, ignore_errors=True)
                        st.toast('The dataset has been deleted!', icon='👍')
                    else:
                        st.warning(f"The dataset {dataset_path} is not available.")

                except Exception as e:
                    st.error(f"Error while deleting the dataset {dataset_path}.")
                    
                    
class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')

import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def DeltaLakeStats():
    st.title(':blue[Instruments Delta Lake] 🧊 📈')
    statType=['Datasets Size stats','Number of Versions stats','Metadata','Schema','History']
    loadStats = st.selectbox(
       "Choose instrument",
       (statType),
       index=None,
    )
    
    
    if loadStats=="Schema":
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
        
        colorNum=0
        for namedir in directories:
            Folderpath="deltalake/"+namedir
            st.title(':green['+namedir+':]')
            dt = DeltaTable(Folderpath)
            st.write(dt.schema())
            
    if loadStats=="History":
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]

        for namedir in directories:
            Folderpath="deltalake/"+namedir
            st.title(':green['+namedir+':]')
            dt = DeltaTable(Folderpath)
            st.write(dt.history()) 
            
    if loadStats=="Metadata":
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]

        for namedir in directories:
            Folderpath="deltalake/"+namedir
            st.title(':green['+namedir+':]')
            st.write(namedir)
            dt = DeltaTable(Folderpath)
            st.write(str(dt.metadata())) 
            
            
            
    
    if loadStats=="Datasets Size stats":
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
        filesizes=[]
        metadata=[]
        colors=[]
        color=['#5A69AF', '#579E65', '#F9C784', '#FC944A', '#F24C00', '#00B825']
           
        colorNum=0
        for namedir in directories:
            Folderpath="deltalake/"+namedir
            size = 0
            # get size
            for path, dirs, files in os.walk(Folderpath):
                for f in files:
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)
            filesizes.append(size)
            metadata.append(str(namedir)+'\n'+str(convert_size(size)))
            colors.append(color[colorNum])
            if colorNum<5:
                colorNum+=1
            else:
                colorNum=0
            
        bubble_chart = BubbleChart(area=filesizes,bubble_spacing=0.1)
        bubble_chart.collapse()
        fig, ax = plt.subplots(figsize=(12,8),subplot_kw=dict(aspect="equal"))
        bubble_chart.plot(ax, metadata,colors)
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()
        ax.set_title('Datasets Size stats')
        st.pyplot(plt)
    
    if loadStats=="Number of Versions stats":
        directories = [entry.name for entry in Path('deltalake').iterdir() if entry.is_dir()]
        versionssizes=[]
        metadata=[]
        colors=[]
        color=['#5A69AF', '#579E65', '#F9C784', '#FC944A', '#F24C00', '#00B825']
            
        colorNum=0
        for namedir in directories:
            Folderpath="deltalake/"+namedir
            dt=DeltaTable(Folderpath)
            versionssizes.append(dt.version())
            metadata.append(str(namedir)+'\n'+str(dt.version()))
            colors.append(color[colorNum])
            if colorNum<5:
                colorNum+=1
            else:
                colorNum=0
            
        bubble_chart = BubbleChart(area=versionssizes,bubble_spacing=0.1)
        bubble_chart.collapse()
        fig, ax = plt.subplots(figsize=(12,8),subplot_kw=dict(aspect="equal"))
        bubble_chart.plot(ax, metadata,colors)
        ax.axis("off")
        ax.relim()
        ax.autoscale_view()
        ax.set_title('Number of Versions stats')
        st.pyplot(plt)
    
    
    

if choice=="🗃️ Home":
     welcome()   
elif choice == "📒 Load ODVs to MERGE in a Dataset":
    reset_dataOdv()    
elif choice=="🧊 📝 Import Data Delta Lake":
     importpageDlake()
elif choice=="🧊 📒 Access Data Delta Lake":
     datareadpageDeltaLake()
elif choice=="🧊 📒 Explore Data Delta Lake":
     dataexplorepageDeltaLake()
elif choice=="🧊 🛠️ Partition Data Delta Lake":
     dataPartitionDeltaLake()
elif choice=="🧊 🦆 SQL on Delta Lake using DuckDB":
     sqlpageduckDeltaLake()
elif choice=="🧊 🗑️ Delete Data on Delta Lake":     
     DeltaLakedeletepage()
elif choice=="🧊 📈 Instruments Delta Lake":     
     DeltaLakeStats()

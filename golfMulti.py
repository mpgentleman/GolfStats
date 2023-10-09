import streamlit as st
import pandas as pd
#import NCAA_Functions as NF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
import requests
import time
from bs4 import BeautifulSoup
import datetime
import time
#from sportsreference.ncaab.teams import Teams
from datetime import datetime
#from sportsreference.ncaab.boxscore import Boxscores
#from sportsreference.ncaab.boxscore import Boxscore
from urllib.request import urlopen
#URL = 'http://classic.sportsbookreview.com/betting-odds/ncaa-basketball/totals/?date=20201229?'
#from selenium import webdriver
#from webdriver_manager.chrome import ChromeDriverManager
import time
#from bs4 import BeautifulSoup
#from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import re
import csv
import os
from itertools import product
#import kenpompy
#from kenpompy.utils import login
#from pyquery import PyQuery as pq
import random
#from joypy import joyplot
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from datetime import datetime
from pytz import timezone
#import schedule
import time
from datetime import datetime, timedelta
from threading import Timer
import random
#from joypy import joyplot
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import pytz
#import gridstatus
def FullDisplay(df):
    with pd.option_context('display.max_rows',10,'display.max_columns',None):
        display(df)
#from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

#from utils.data import DateInfo
#from utils.field_config import FIELD_CONFIG

def cellStyleDynamic(data: pd.Series):

    datNeg = data[data < 0]
    datPos = data[data > 0]

    if len(datNeg) > 0 and len(datPos) > 0:
        _, binsN = pd.cut(datNeg, bins=4, retbins=True, precision=0)
        _, binsP = pd.cut(datPos, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < %d) return {'color': 'white','backgroundColor':  '#ff0000'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff4c4c'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff9999'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#a8bad9'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#7d97c6'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#3c64aa'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (binsN[1], binsN[2], binsN[3], binsP[1], binsP[2], binsP[3])

    elif len(datNeg) > 0:
        dat, bins = pd.cut(datNeg, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < %d) return {'color': 'white','backgroundColor':  '#ff0000'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff4c4c'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#ff9999'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (bins[1], bins[2], bins[3])

    elif len(datPos) > 0:
        dat, bins = pd.cut(datPos, bins=4, retbins=True, precision=0)
        code = """
            function(params) {
              if (isNaN(params.value) || params.value === 0) return {'color': 'black', 'backgroundColor': '#e9edf5'};
              if (params.value < 0) return {'color': 'white', 'backgroundColor': '#ffb2b2'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#a8bad9'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#7d97c6'};
              if (params.value < %d) return {'color': 'white', 'backgroundColor': '#3c64aa'};
              
              return {'color': 'white', 'backgroundColor': '#2753a1'};
            };
            """ % (bins[1], bins[2], bins[3])
    else:
        code = """
            function(params) {
              return {'color': 'white', 'backgroundColor': '#a8bad9'};
            };
            """

    return JsCode(code)
cellStyle = JsCode("""
            function(params) {
                return {'color': 'black','backgroundColor':  '#e8f4f8'};
            };
            """)
agContextMenuItemsDeluxe = JsCode(
    """
    function getContextMenuItems(params) {
      const result = [
        'copy',
        'copyWithHeaders',
        'paste',
        'separator',
        'autoSizeAll',
        'expandAll',
        'contractAll',
        'resetColumns',
        'separator',
        'export',
      ];
      
      return result;
    }
    """
)

agContextMenuItemsBasic = JsCode(
    """
    function getContextMenuItems(params) {
      const result = [
        'copy',
        'copyWithHeaders',
        'paste',
        'separator',
        'autoSizeAll',
        'resetColumns',
        'separator',
        'export',
      ];
      
      return result;
    }
    """
)


_type_mapper = {
    "b": ["textColumn"],
    "i": ["numericColumn", "numberColumnFilter"],
    "u": ["numericColumn", "numberColumnFilter"],
    "f": ["numericColumn", "numberColumnFilter"],
    "c": [],
    "m": ['timedeltaFormat'],
    "M": ["dateColumnFilter", "customDateTimeFormat"],
    "O": [],
    "S": [],
    "U": [],
    "V": [],
}



DEFAULT_COL_PARAMS = dict(
    filterParams=dict(buttons=['apply', 'reset'], closeOnApply=True),
    groupable=False,
    enableValue=True,
    enableRowGroup=True,
    enablePivot=False,
    editable=False
)


DEFAULT_STATUS_PANELS = [
    dict(statusPanel='agTotalAndFilteredRowCountComponent', align='left'),
    dict(statusPanel='agAggregationComponent',
         # statusPanelParams={
         #     'aggFuncs': ['count', 'sum', 'min', 'max', 'avg']
         # },
         statusPanelParams=dict(aggFuncs=['count', 'min', 'max', 'sum']),
         align='right',
         ),
]

DEFAULT_STATUS_BAR = {
    'statusPanels': DEFAULT_STATUS_PANELS
}


def gridOptionsFromDataFrame(df: pd.DataFrame, field_config=None, **default_column_parameters) -> GridOptionsBuilder:

    gb = GridOptionsBuilder()

    params = {**DEFAULT_COL_PARAMS, **default_column_parameters}
    gb.configure_default_column(**params)

    if any('.' in col for col in df.columns):
        gb.configure_grid_options(suppressFieldDotNotation=True)

    # fconfig = field_config if field_config else FIELD_CONFIG
    #fconfig = field_config or FIELD_CONFIG
    fconfig = field_config

    for col, col_type in zip(df.columns, df.dtypes):
        if col in fconfig:
            conf = fconfig.get(col)
            if 'type' not in conf:
                gb.configure_column(field=col, type=_type_mapper.get(col_type.kind, []), **conf)
            else:
                gb.configure_column(field=col, **conf)
        else:
            gb.configure_column(field=col, type=_type_mapper.get(col_type.kind, []))

    return gb


def createCsvExportParams(name: str, DateInfo):
    n = name.replace(' ', '_')
    fname = f'{n}_{DateInfo}.csv'
    return {'defaultCsvExportParams': {'fileName': fname}}


def createExportParams(name: str, dt: datetime, csv=True, excel=True):
    n = name.replace(' ', '_')
    sdt = dt.strftime('%Y%m%d_%H%M%S')
    fname = f'{n}_{sdt}'
    params = {}
    if csv:
        params['defaultCsvExportParams'] = {'fileName': f'{fname}.csv'}
    if excel:
        params['defaultExcelExportParams'] = {'fileName': f'{fname}.xlsx', 'sheetName': 'Sheet1'}

    return params


DEFAULT_GRID_OPTIONS = dict(
    domLayout='normal',
    # rowGroupPanelShow='always',
    statusBar=DEFAULT_STATUS_BAR,
    autoGroupColumnDef=dict(pinned='left'),
    getContextMenuItems=agContextMenuItemsBasic,
    suppressAggFuncInHeader=True,
    # pivotPanelShow='onlyWhenPivoting',
    pivotPanelShow='always',
    # pivotMode=False,
    # pivotPanelShow='always',
    # pivotColumnGroupTotals='before',
    # rowSelection='multiple',
    enableRangeSelection=True,
    suppressMultiRangeSelection=False,
    # defaultCsvExportParams=dict(fileName='testExport.csv'),
    suppressExcelExport=False,
)


CUSTOM_CSS = {
    '.ag-theme-streamlit': {
        'font-family': ("Consolas", 'monospace'),
        # 'grid-size': '2px',
    },
    '.ag-theme-light': {
        'font-family': ("Consolas", 'monospace'),
        # 'grid-size': '2px',
    }
}


def displayGrid(df: pd.DataFrame, key: str,
                reloadData: bool, updateMode: GridUpdateMode=GridUpdateMode.VALUE_CHANGED,
                customCss=None):

    gb = gridOptionsFromDataFrame(df)
    gb.configure_side_bar()
    gb.configure_grid_options(**DEFAULT_GRID_OPTIONS)
    gridOptions = gb.build()

    # updateMode = GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.VALUE_CHANGED
    dataReturnMode = DataReturnMode.FILTERED_AND_SORTED

    g = AgGrid(df, gridOptions=gridOptions, height=700, key=key, editable=False,
               enable_enterprise_modules=True,
               allow_unsafe_jscode=True,
               fit_columns_on_grid_load=False,
               reload_data=reloadData,
               # theme='streamlit',
               update_mode=updateMode,
               data_return_mode=dataReturnMode,
               custom_css=customCss
               )

    return g


def displayGrid2(df: pd.DataFrame, gb: GridOptionsBuilder,
                 key: str, reloadData: bool=False, updateMode: GridUpdateMode=GridUpdateMode.VALUE_CHANGED,
                 height=700, fit_columns_on_grid_load=True, editable=False, customCss=None):

    gridOptions = gb.build()

    # updateMode = GridUpdateMode.FILTERING_CHANGED | GridUpdateMode.VALUE_CHANGED
    dataReturnMode = DataReturnMode.FILTERED_AND_SORTED

    g = AgGrid(df, gridOptions=gridOptions, height=height, key=key, editable=editable,
               enable_enterprise_modules=True,
               allow_unsafe_jscode=True,
               fit_columns_on_grid_load=fit_columns_on_grid_load,
               reload_data=reloadData,
               # theme='streamlit',
               update_mode=updateMode,
               data_return_mode=dataReturnMode,
               custom_css=customCss
               )

    return g




def plot_circle(x):
    # create a figure and an axes object
    fig, ax = plt.subplots()
    # create a circle object with center (0, 0) and radius x
    circle = plt.Circle((0, 0), x, color='lightgreen')
    # add the circle to the axes
    ax.add_patch(circle)
    # set the aspect ratio of the axes to 1
    ax.set_aspect(1)
    # show the plot
    plt.show()
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def create_model(df, x_col, y_col):
  # Create and fit the model using the dataframe and the column names
  X = df[x_col].values.reshape(-1, 1) # Features
  y = df[y_col].values # Target
  model = LinearRegression()
  model.fit(X, y)
  return model # Return the model object

# Define the prediction function
def add_prediction(df, x_col, model):
  # Make predictions using the dataframe, the column name, and the model object
  X = df[x_col].values.reshape(-1, 1) # Features
  y_pred = model.predict(X) # Predictions
  df['Avg Shots Pred'] = y_pred # Add a new column with the predictions
  return df # Return the updated dataframe



# Import the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Define the model function
def create_model_poly(df, x_col, y_col, degree):
  # Create and fit the model using the dataframe, the column names, and the degree of the polynomial
  X = df[x_col].values.reshape(-1, 1) # Features
  y = df[y_col].values # Target
  poly = PolynomialFeatures(degree) # Polynomial transformer
  X_poly = poly.fit_transform(X) # Transform features into polynomial terms
  model = LinearRegression()
  model.fit(X_poly, y)
  return model, poly # Return the model and the transformer objects

# Define the prediction function
def add_prediction_poly(df, x_col, model, poly):
  # Make predictions using the dataframe, the column name, the model object, and the transformer object
  X = df[x_col].values.reshape(-1, 1) # Features
  X_poly = poly.transform(X) # Transform features into polynomial terms
  y_pred = model.predict(X_poly) # Predictions
  df['Avg Shots Pred'] = y_pred # Add a new column with the predictions
  return df # Return the updated dataframe



def plot_point_cov2(points,width_of_green, distance_to_green,nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    # plot the ellipse
    fig, ax = plt.subplots()
    if ax is None:
        ax = plt.gca()
    fig = plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
    # create a circle object with radius x and color lightgreen
    circle = plt.Circle((0, distance_to_green), width_of_green, color='lightgreen')
    # add the circle to the same axes as the ellipse
    ax.add_patch(circle)
    plt.show()
    st.pyplot(fig)

    # return both the ellipse and the circle
    return ellipse, circle



class Practice_Session():
    """

    """

    def __init__(self, filename,club_selection,distance): # this is the constructor!
        '''
        Initialize 
        :param 
        :param 
        :return:
        '''
        self._data = filename
        #st.dataframe(self._data)
        #self._data = self._data[self._data['Club Type']==club_selection]
        self._distance = distance
        self.calc_df()
        
    def calc_df(self):
        shots_input = pd.read_csv('folder/Strokes_to_Hole.csv')
        avg_strokes = pd.read_csv('folder/Avg_Strokes.csv')
        model, poly = create_model_poly(shots_input, 'Feet', 'Avg Shots',2)
        model2 = create_model(avg_strokes, 'Yards', 'Strokes_To_Hole')
        
        self._data['Yards_off_center'] = ((self._data['Total Distance']-self._distance)*(self._data['Total Distance']-self._distance)+self._data['Total Lateral Distance']*self._data['Total Lateral Distance'])
        self._data['Feet_from_Pin'] = self._data[['Yards_off_center']].sum(axis=1).pow(1./2)*3
        self._data = add_prediction_poly(self._data, 'Feet_from_Pin', model,poly)
        self._data['Strokes_Gained'] =  model2.predict([[143]])[0] - self._data['Avg Shots Pred'] -1
        self.avg_distance = self._data['Total Distance'].mean()
        self.lateral_distance = self._data['Total Lateral Distance'].mean()
        self.strokes_gained = self._data['Strokes_Gained'].mean()
        col1, col2 = st.columns(2)
        with col1:
             self.print_data()
        with col2:
            fig, ax = plt.subplots()

            fig =self.show_green_shots()

            # display the chart in Streamlit
            st.pyplot(fig)
        
            #FullDisplay(self._data)
            #st.dataframe(self._data)
    def print_data(self):
        #st.write('Average Distance was ',self.avg_distance)
        st.metric(label ='Average Distance was ',value = round(self.avg_distance,2))
        st.metric(label ='Total Lateral Distance was ',value =round(self.lateral_distance,2))
        st.metric(label ='Avg Strokes Gained ',value =round(self.strokes_gained,3))
    
    def plot_point_cov2(self,points,width_of_green, distance_to_green,nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma ellipse based on the mean and covariance of a point
        "cloud" (points, an Nx2 array).

        Parameters
        ----------
            points : An Nx2 array of the data points.
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """
        pos = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        # plot the ellipse
        fig, ax = plt.subplots()
        if ax is None:
            ax = plt.gca()
        ellipse = plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
        # create a circle object with radius x and color lightgreen
        #circle = plt.Circle((0, distance_to_green), width_of_green, color='lightgreen', alpha=.5)
        #circle = plt.Circle((0, distance_to_green), 10, color='lightgreen', alpha=.5)
        # add the circle to the same axes as the ellipse
        ax.add_patch(circle)
        #st.pyplot(fig)
        #st.pyplot(ellipse)
        #plt.show()
        # return both the ellipse and the circle
        return ellipse, circle
    def show_green_shots(self,ax=None, **kwargs):
        sns.set_style("ticks",{'axes.grid' : True})
        fig, ax = plt.subplots()
        #ax.plot(self._data['Total Lateral Distance'], self._data['Total Distance'], 'ro')
        
        C = np.array(list(map(list, zip(self._data.loc[:, 'Total Lateral Distance'], self._data.loc[:, 'Total Distance']))))
        points = C
        pos = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        ellipse = plot_cov_ellipse(cov, pos, 10, ax, alpha=1, color='lightblue')
        #st.write(C)
        
        #circle = plt.Circle((0,self._distance ),20, color='lightgreen', alpha=.5)
        #ax.add_patch(circle)

        #self.plot_point_cov2(C, 10,self._distance,nstd=2, alpha=.5, color='lightblue')
        #st.pyplot(fig)
        
        fig = sns.jointplot(data=self._data, x="Total Lateral Distance", y="Total Distance",hue='Club Type', height=6)
        circle = plt.Circle((0,self._distance ),10, color='lightgreen', alpha=.5)
        ax.add_patch(circle)
        #st.pyplot(fig)
        #fig = sns.jointplot(data=self._data, x="Total Lateral Distance", y="Total Distance",hue='Strokes_Gained', height=6)
        #fig.ax_joint.plot([0],[self._distance],'o',ms=100 , mec='r', mfc='none')
        fig.ax_joint.plot([0],[self._distance],'o',ms=100 , mec='r', mfc='none')
        
        return fig

class Practice_SessionMulti():
    """

    """

    def __init__(self, filename): # this is the constructor!
        '''
        Initialize 
        :param 
        :param 
        :return:
        '''
        self._data = filename
        #st.dataframe(self._data)
        #self._data = self._data[self._data['Club Type']==club_selection]
        #self._distance = distance
        self.calc_df()
        
    def calc_df(self):
        shots_input = pd.read_csv('folder/Strokes_to_Hole.csv')
        avg_strokes = pd.read_csv('folder/Avg_Strokes.csv')
        self.avg_distance = self._data['Total Distance'].mean()
        self.lateral_distance = self._data['Total Lateral Distance'].mean()
    def print_data(self):
        #st.write('Average Distance was ',self.avg_distance)
        st.metric(label ='Average Distance was ',value = round(self.avg_distance,2)),
        st.metric(label ='Total Lateral Distance was ',value =round(self.lateral_distance,2))
        #st.metric(label ='Avg Strokes Gained ',value =round(self.strokes_gained,3))
    
    
def convert_columns_to_float(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
def filter_row(df, column, value):
    filtered_df = df[df[column] == value]
    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        return row.to_dict()
    else:
        return None


import plotly.graph_objects as go
def plot_values(result):
    keysList = list(result.keys())
    title1 = keysList[1]
   
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
    mode = "number+gauge", value = result['Club Speed'],
    domain = {'row': 0, 'column': 1},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Club Speed</b><br><span style='color: gray; font-size:0.8em'>mph</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [50, 150]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Club Speed']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Club Speed'].min(), myData._data['Club Speed'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))

    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Attack Angle'],
    domain = {'row': 0, 'column': 2},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Attack Angle</b><br><span style='color: gray; font-size:0.8em'>degrees</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [-20, 20]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Attack Angle']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Attack Angle'].min(), myData._data['Attack Angle'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))

    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Smash'],
    domain = {'row': 1, 'column': 2},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Smash</b><br><span style='color: gray; font-size:0.8em'>ratio</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [.5, 2]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Smash']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Smash'].min(), myData._data['Smash'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))

    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Spin Rate'],
    domain = {'row': 1, 'column': 1},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Spin Rate</b><br><span style='color: gray; font-size:0.8em'>mph</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 10000]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Spin Rate']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Spin Rate'].min(), myData._data['Spin Rate'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))
    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Ball Speed'],
    domain = {'row': 2, 'column': 2},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Ball Speed</b><br><span style='color: gray; font-size:0.8em'>mph</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [50, 150]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Ball Speed']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Ball Speed'].min(), myData._data['Ball Speed'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))

    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Vertical Launch'],
    domain = {'row': 2, 'column': 1},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Vertical Launch</b><br><span style='color: gray; font-size:0.8em'>degrees</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 60]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Vertical Launch']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Vertical Launch'].min(), myData._data['Vertical Launch'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))
    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Peak Height'],
    domain = {'row': 3, 'column': 2},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Peak Height</b><br><span style='color: gray; font-size:0.8em'>ft</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [50, 150]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Peak Height']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Peak Height'].min(), myData._data['Peak Height'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))

    fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  result['Total Distance'],
    domain = {'row': 3, 'column': 1},
    #delta = {'reference': 280, 'position': "top"},
    title = {'text':"<b>Total Distance</b><br><span style='color: gray; font-size:.8em'>yards</span>", 'font': {"size": 8}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [50, 280]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': result['Total Distance']},
        'bgcolor': "white",
        'steps': [
            
            {'range': [myData._data['Total Distance'].min(), myData._data['Total Distance'].max()], 'color': "lightgreen"}],
        'bar': {'color': "rgba(0,0,0,0)"}
    }
))


    fig.update_layout(
    grid = {'rows': 4, 'columns': 3, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {'text': "Speed"},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': 90}}]
                         }}
)
    fig.update_layout(height=800,width=1000)
    st.plotly_chart(fig)


def vertical_mean_line_survived2(x, **kwargs):
    ls = {"0":"-","1":"--"}
    plt.axvline(x.mean(), linestyle =ls[kwargs.get("label","0")], 
                color = kwargs.get("color", "g"))
    #txkw = dict(size=18, color = kwargs.get("color", "g"), rotation=90)
    txkw = dict(size=18, color = kwargs.get("color", "g"))
    #tx = "mean: {:.2f}, std: {:.2f}".format(x.mean(),x.std())
    tx = "mean: {:.2f}".format(x.mean())
    plt.text(x.mean(), 0, tx, **txkw)


from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


def get_Histogram_Facet(df):
    tall_df = df.melt(var_name='Measurement', value_name='Value')
    sns.set_theme()
    grid = sns.displot(kind='kde', data=tall_df, col='Measurement', col_wrap=4, x='Value', hue='Measurement', fill=True, facet_kws={'sharey': False, 'sharex': False})
    grid.map(vertical_mean_line_survived2, 'Value')
    with _lock:
        st.pyplot(grid)

st.set_page_config(
    page_title="Golf App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.header("Golf Analysis App")

page = st.sidebar.selectbox('Select page',['One Club','Club Distances']) 

if page == 'One Club':
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False
        
        else:
            st.write('Please upload a file')
        
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Drop the second row (index 1)
        df = df.drop(df.index[0])
        club_selected = list(df['Club Type'].unique())
        option = st.selectbox('Which Club?',club_selected)
        df=df[df['Club Type']==option]
        convert_column_names = ['Total Distance','Total Lateral Distance']
        colsM=['Club Speed','Ball Speed','Carry Distance','Total Distance',
                   'Roll Distance','Smash','Vertical Launch','Peak Height',
                   'Descent Angle','Horizontal Launch','Carry Lateral Distance',
                   'Total Lateral Distance','Carry Curve Distance','Total Curve Distance',
                   'Attack Angle','Dynamic Loft','Spin Loft','Spin Rate','Spin Axis',
                   'Club Path','Face Path','Face Target'
               ]
        df = convert_columns_to_float(df, colsM)
        st.write('You selected:', option)
        text_input = st.number_input("Enter yardage 👇",) 
    

        # Display the modified data



                
    


        #col1, col2 = st.columns(2)
        filename = df
        club = club_selected
        Pin_in_yards = text_input
        st.subheader('Golf Metrics Dashboard')    
        #with col1:
        myData = Practice_Session(filename,club,Pin_in_yards)
     


    
        team1 = myData._data
        colsM=['Club Speed','Ball Speed','Carry Distance','Total Distance','Roll Distance',
                   'Smash','Vertical Launch','Peak Height','Descent Angle',
                   'Horizontal Launch','Carry Lateral Distance','Total Lateral Distance',
                   'Carry Curve Distance','Total Curve Distance','Attack Angle','Dynamic Loft',
                   'Spin Loft','Spin Rate','Spin Axis','Club Path','Face Path','Face Target',
                   'Shot Classification','Feet_from_Pin','Strokes_Gained'
                    ]
        colsM2=['Club Speed','Ball Speed','Carry Distance','Total Distance','Roll Distance',
                    'Smash','Vertical Launch','Peak Height','Descent Angle',
                    'Horizontal Launch','Carry Lateral Distance','Total Lateral Distance','Carry Curve Distance',
                    'Total Curve Distance','Attack Angle','Dynamic Loft',
                    'Spin Loft','Spin Rate','Spin Axis','Club Path','Face Path','Face Target','Feet_from_Pin','Strokes_Gained'
                     ]
        numeric=['numericColumn','numberColumnFilter']

        team2=team1[colsM]
        team3=team1[colsM2]
        colsDist=['Carry Distance','Total Distance','Roll Distance','Total Lateral Distance']
        colsSpeed=['Club Speed','Ball Speed','Smash']
        colsPath=['Spin Axis','Club Path','Face Path','Face Target']
        colsSpin=['Vertical Launch','Peak Height','Descent Angle', 'Horizontal Launch',
                      'Attack Angle','Dynamic Loft','Spin Loft','Spin Rate']
        colsStrokes=['Feet_from_Pin','Strokes_Gained']
        st.subheader('Distance Metrics')
        get_Histogram_Facet(team3[colsDist])
        st.subheader('Speed Metrics')
        get_Histogram_Facet(team3[colsSpeed])
        st.subheader('Club Path Metrics')
        get_Histogram_Facet(team3[colsPath])
        st.subheader('Spin Metrics')
        get_Histogram_Facet(team3[colsSpin])
        st.subheader('Green Metrics')
        get_Histogram_Facet(team3[colsStrokes])
        allcols=team2.columns
        header1=option+' Shot Data'
        st.subheader(header1)
        csTotal=cellStyleDynamic(team1.Strokes_Gained)
        gb = GridOptionsBuilder.from_dataframe(team2)
        #gb.configure_columns('PlayingOverRating', type=numeric, valueFormatter=numberFormat(1))
        gb.configure_columns(allcols, cellStyle=cellStyle)
        #gb.configure_column('PlayingOverRating',cellStyle=csTotal,valueFormatter=numberFormat(1))
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()

        AgGrid(team2, gridOptions=gridOptions, enable_enterprise_modules=True,height=500,allow_unsafe_jscode=True)
        IdealShot= pd.read_csv('folder/IdealTrackman.csv')
        #st.dataframe(IdealShot)
        result = filter_row(IdealShot,'Club',option)
        st.subheader(' Ideal Zones. Green is the range from this session')
        plot_values(result)
else:
    st.write('Golf Multi')
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Drop the second row (index 1)
            df = df.drop(df.index[0])
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False
    



        
        #option = st.selectbox('Which Club?',club_selected)
        #df=df[df['Club Type']==option]
        convert_column_names = ['Total Distance','Total Lateral Distance']
        colsM=['Club Speed','Ball Speed','Carry Distance','Total Distance','Roll Distance','Smash','Vertical Launch','Peak Height','Descent Angle',
           'Horizontal Launch','Carry Lateral Distance','Total Lateral Distance','Carry Curve Distance','Total Curve Distance','Attack Angle','Dynamic Loft',
           'Spin Loft','Spin Rate','Spin Axis','Club Path','Face Path','Face Target'
           ]
        df = convert_columns_to_float(df, colsM)
        #st.write('You selected:', option)
        #text_input = st.number_input("Enter yardage 👇",) 
    

        # Display the modified data    
    


    #col1, col2 = st.columns(2)
    filename = df
    #club = club_selected
    #Pin_in_yards = text_input
    st.subheader('Golf Metrics Dashboard')    
    #with col1:
    myData = Practice_SessionMulti(filename)
     


    
    team1 = myData._data
    club_selected = list(team1['Club Type'].unique())
    #st.write(club_selected)
   # st.dataframe(team1)

    colsM=['Club Speed','Ball Speed','Carry Distance','Total Distance','Roll Distance','Smash','Vertical Launch','Peak Height','Descent Angle',
           'Horizontal Launch','Carry Lateral Distance','Total Lateral Distance','Carry Curve Distance','Total Curve Distance','Attack Angle','Dynamic Loft',
           'Spin Loft','Spin Rate','Spin Axis','Club Path','Face Path','Face Target','Shot Classification','Feet_from_Pin','Strokes_Gained'
           ]
    colsM2=['Club Speed','Ball Speed','Carry Distance','Total Distance','Total Lateral Distance','Roll Distance'
           ]
    numeric=['numericColumn','numberColumnFilter']
    for i in range(len(club_selected)):
        st.subheader(club_selected[i]+' Distance Metrics')
        team2=team1[team1['Club Type']==club_selected[i]]
        #team2=team1[colsM]
        team3=team2[colsM2]
        #st.dataframe(team3)
        colsDist=['Carry Distance','Total Distance','Roll Distance','Total Lateral Distance']
        colsSpeed=['Club Speed','Ball Speed','Smash']
        #st.subheader('Distance Metrics')
        get_Histogram_Facet(team3[colsDist])

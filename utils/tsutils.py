import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from cycler import cycler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from . import api_key

A = plt.cm.Set3(np.linspace(0,1,15))
plt.rcParams["axes.prop_cycle"] = cycler('color', [ rgb2hex(A[i,:]) for i in range(A.shape[0]) ])


PRESIDENT_INAUGURATION = [
    ('Truman','Democrat','1945-04-12'),
    ('Eisenhower','Republican','1953-01-20'),
    ('Kennedy','Democrat','1961-01-20'),
    ('Lyndon Johnson','Democrat','1963-11-22'),
    ('Nixon','Republican','1969-01-20'),
    ('Ford','Republican','1974-08-09'),
    ('Carter','Democrat','1977-01-20'),
    ('Reagan','Republican','1981-01-20'),
    ('H.W. Bush','Republican','1989-01-20'),
    ('Clinton','Democrat','1993-01-20'),
    ('W. Bush','Republican','2001-01-20'),
    ('Obama','Democrat','2009-01-20'),
    ('Trump','Republican','2017-01-20'),
    ('Biden','Democrat','2021-01-20')]

PRESIDENT_INAUGURATION = (pd.DataFrame(
    PRESIDENT_INAUGURATION,
    columns=["president","party","inaug_date"])
    .assign(inaug_date = lambda df: pd.to_datetime(df.inaug_date))
)

LABEL_MAP = {
    "real_gdp" : "Real GDP",
    "tot_non_farm_empl" : "Total Non-Farm Employment",
    "manufact_empl" : "Manufacturing Employment",
    "unemploy_rate" : "Unemployment Rate",
    "national_debt" : "National Debt",
    "recession" : "Reccessions",
    "pers_hlthcr_exp" : "Personal Healthcare Expenditures",
    "real_grss_dom_prv_inv" : "Real Gross Domestic Private Investment",
    "fed_gov_curr_tax_receipts" : "Federal Gov't Current Tax Receipts",
    "fed_gov_curr_exp" : "Federal Gov't Current Expenditures",
    "cpi" : "Consumer Price Index"
}

def get_fred_data(x = "all"):
    """
    Get data from the St Louis FED

    Parameters
    ----------
    x : str
        name of series to get

    Return
    ------
    res : dict
        dictionary of results
    """
    fred = Fred(api_key = api_key.fred_api_key)
    if x == 'all':
        res = {}
        res['real_gdp'] = {
            'data':fred.get_series('GDPC1'),
            'name': 'Real GDP',
            'units': 'Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate'}
        res['tot_non_farm_empl'] = {
            'data':fred.get_series('PAYEMS'),
            'name': 'Total Nonfarm Employees',
            'units': 'Thousands of Persons, Seasonally Adjusted'}
        res['manufact_empl'] = {
            'data':fred.get_series('MANEMP'),
            'name': 'Manufacturing Employees',
            'units': 'Thousands of Persons, Seasonally Adjusted'}
        res['unemploy_rate'] = {
            'data':fred.get_series('UNRATE'),
            'name': 'Unemployment Rate',
            'units' : 'Percent, Seasonally Adjusted'}
        res['national_debt'] = {
            'data': fred.get_series("GFDEBTN"),
            'name': 'Total Federal Debt',
            'units': 'Millions of Dollars, Not Seasonally Adjusted'}
        res['recession'] = {
            'data': fred.get_series('USRECQ'),
            'name': 'Recessions',
            'units': None}
        res['pers_hlthcr_exp'] = {
            'data': fred.get_series("DHLCRC1Q027SBEA"),
            'name': 'Personal Healthcare Expenditures',
            'units': 'Billions of Dollars, Seasonally Adjusted Annual Rate'}
        res['real_grss_dom_prv_inv'] = {
            'data': fred.get_series("GPDIC1"),
            'name': 'Real Gross Private Domestic Investment',
            'units': 'Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate'}
        res['fed_gov_curr_tax_receipts'] = {
            'data': fred.get_series("W006RC1Q027SBEA"),
            'name': 'Federal Govt Current Tax Receipts',
            'units': 'Billions of Dollars, Seasonally Adjusted Annual Rate'}
        res['fed_gov_curr_exp'] = {
            'data':fred.get_series("NA000283Q"),
            'name': 'Federal Govt Current Expenditures',
            'units': 'Millions of Dollars, Not Seasonally Adjusted'}
        res['cpi'] = {
            'data': fred.get_series("CPALTT01USQ661S"),
            'name': 'Consumer Price Index',
            'units': 'Index 2015=100, Seasonally Adjusted'}
        return res

def write_data_to_json(data, file = None):
    """
    Write data to a JSON document

    Parameters
    ----------
    data : dict of pandas.Series

    file : str
        path to JSON file

    Returns
    -------
    None
    """
    for k in data.keys():
        data[k]['data'].index = data[k]['data'].index.astype(str)
        data[k]['data'] = data[k]['data'].to_dict()
    if file is None:
        file = Path(__file__).parents[1] / 'data' / 'fred_data.json'
    with open(file,"w") as f:
        json.dump(data,f)


def read_data_from_json(file = None):
    if file is None:
        file = Path(__file__).parents[1] / 'data' / 'fred_data.json'
    with open(file,"r") as f:
        data = json.load(f)
    for k in data.keys():
        data[k]['data'] = pd.Series(data[k]['data'])
        data[k]['data'].index = pd.to_datetime(data[k]['data'].index)
        data[k]['data'] = (
            pd.merge_asof(
                data[k]['data'].to_frame(name=k)
                    .reset_index()
                    .rename(columns={"index":"date"}),
                PRESIDENT_INAUGURATION,
                left_on="date",right_on="inaug_date"))
    return data

def get_recession_intervals(recession):
    fix_rec = lambda df, col: np.maximum(0,df[col].fillna(df.recession))

    recession = (
        recession['data'].dropna()
        .assign(begin_recession = lambda df: (df.recession - df.recession.shift()))
        .assign(begin_recession = lambda df: fix_rec(df,"begin_recession"))
        .assign(end_recession = lambda df: (df.recession - df.recession.shift(-1)))
        .assign(end_recession = lambda df: fix_rec(df,"end_recession")))
    begin_recession = recession.query("begin_recession == 1").date.to_list()
    end_recession = recession.query("end_recession == 1").date.to_list()
    end_recession = [i + relativedelta(months=+3) for i in end_recession]

    return list(zip(begin_recession,end_recession))


def remove_borders(ax):
    for x in ["top","right"]:
        ax.spines[x].set_visible(False)

def plot_all_plotly(data):
    ts = list(data.keys())
    ts.remove("recession")
    recessions = get_recession_intervals(data['recession'])
    fig = make_subplots(
        rows = (len(ts) + 1) // 2,
        cols=2,
        subplot_titles=[
            data[k]['name'] + '<br><span style="font-size: 12px;">' + data[k]['units'] + "</span>"
            for k in ts])
    for i, k in enumerate(ts):
        r,c = 1 + (i//2), 1 + (i%2)
        fig.add_trace(go.Scatter(
            x=data[k]['data'].date,
            y=data[k]['data'][k],
            showlegend=False,
            name="",
            hovertemplate = "Date: %{x}<br>" + data[k]['name'] + ": %{y}"),
            row=r,
            col=c)
        fig['layout'][f'yaxis{str(i+1)}']['title'] = data[k]['name']
        fig['layout'][f'xaxis{str(i+1)}']['title'] = 'Date'
        for rec in recessions:
            fig.add_vrect(x0=rec[0],x1=rec[1], line_width=0, opacity=0.1, fillcolor='red',row=r, col=c)
    return fig


def _freq(data):
    n = data.loc[lambda df: (df.date >= '2000-01-01') & (df.date < '2020-01-01')].shape[0]
    if n == 20:
        return "Y"
    elif n == 80:
        return "Q"
    elif n == 240:
        return "M"
    else:
        raise ValueError(f"There are {n} observations in the timespan" + 
        " 2000-01-01 to 2020-01-01, should be in 20, 80, 240")


def plot_change(data, recessions, transform='gmean'):
    feature = data['data'].columns[1]
    freq = _freq(data['data'])
    if freq == "Q" : z=4
    elif freq == "M": z=12
    elif freq == "Y": z=1

    if transform == "gmean":
        transform = gmean
    elif transform == "mean":
        transform = np.mean
    elif transform == "median":
        transform = np.median
    data_for_plotting = (data['data']
        .assign(delta = lambda df: df[feature] / df[feature].shift())
        .dropna()
        .assign(perc = lambda df: df.groupby('president')['delta'].transform(transform))
        .assign(delta_an = lambda df: 100*df["delta"].map(lambda x: x**z - 1))
        .assign(perc = lambda df: 100*df["perc"].map(lambda x: x**z - 1)))
    fig = px.line(data_for_plotting,x='date',y='delta_an',color='president',
                  hover_data=['date','president','delta_an'],
                  labels={'date':"Date","president":"President",
                          "delta_an":f'Annualized Change in {data["name"]}<br>by Administration',
                          "perc":f'Annualized Change in {data["name"]}<br>by Administration'},
                 title=f'Annualized Change in {data["name"]} by Administration<br><span style="font-size: 12px;">{data["units"]}</span>')
    for d in fig.data:
        temp = data_for_plotting.query(f"president == '{d.name}'").sort_values('date')
        #temp = pd.concat([temp.head(1),temp.tail(1)])
        fig.add_trace(go.Scatter(
            x=temp['date'],
            y=temp['perc'],
            line={'color':d.line.color},
            mode='lines',
            showlegend=False,
            name="",
            hovertemplate = "President: " + d.name + "<br>Administration Annualized Change: %{y}",
            legendgroup = d.legendgroup))
    for r in recessions:
        fig.add_vrect(x0=r[0],x1=r[1], line_width=0, opacity=0.1, fillcolor='red')
    return fig

def pres_bar_plotly(data, transform="gmean"):
    feature = data['data'].columns[1]
    freq = _freq(data['data'])
    if freq == "Q" : z=4
    elif freq == "M": z=12
    elif freq == "Y": z=1

    if transform == "gmean":
        transform = gmean
    elif transform == "mean":
        transform = np.mean
    elif transform == "median":
        transform = np.median
    summary_data = (data['data']
        .assign(delta = lambda df: df[feature] / df[feature].shift())
        .dropna()
        .assign(perc = lambda df: df.groupby("president")['delta'].transform(transform))
        .assign(delta_an = lambda df: df["delta"].map(lambda x: x**z - 1))
        .assign(perc = lambda df: round(100*df["perc"].map(lambda x: x**z - 1),1)))
    summary_data = (summary_data
        .loc[:,["president","perc","party"]]
        #.assign(color = lambda df: df.party.map(lambda x: 'blue' if x=='Democrat' else 'red'))
        .drop_duplicates()
        .sort_values('perc'))
    
    title = data['name']
    fig = px.bar(summary_data,x="president",y='perc',color='party',
        color_discrete_map={'Republican':'red','Democrat':'blue'},
        title = f'Percent Average Annual Change in {title}<br><span style="font-size: 12px;">by President</span>',
        labels={'party':'Party','president':'President','perc':f'Percent Average Annual Change in<br>{title}'})
    fig.update_xaxes(categoryorder='array', categoryarray= summary_data['president'].tolist())
    return fig

# def plot_all(data):
#     ts = list(data.keys())
#     ts.remove("recession")
#     recessions = get_recession_intervals(data['recession'])
#     fig, axes = plt.subplots((len(ts) + 1) // 2,2,figsize = (12,2*len(ts)));
#     axesf = axes.flatten();
#     for i, k in enumerate(ts):
#         axesf[i].plot(data[k].date,data[k][k]);
#         axesf[i].set_ylabel(LABEL_MAP.get(k,k));
#         remove_borders(axesf[i])
#         for x in recessions:
#             if x[0] >= pd.to_datetime('1939-01-01'):
#                 axesf[i].axvspan(x[0], x[1], alpha=0.05, color='grey');
#     fig.tight_layout();
#     return fig

# def pres_bar_plot(data, x, freq="Q",transform="gmean"):
#     if freq == "Q" : z=4
#     elif freq == "M": z=12
#     elif freq == "Y": z=1

#     if transform == "gmean":
#         transform = gmean
#     elif transform == "mean":
#         transform = np.mean
#     elif transform == "median":
#         transform = np.median
#     zzz = (data[x]
#         .assign(delta = lambda df: df[x] / df[x].shift())
#         .dropna()
#         .assign(perc = lambda df: df.groupby("president")['delta'].transform(transform))
#         .assign(delta_an = lambda df: df["delta"].map(lambda x: x**z - 1))
#         .assign(perc = lambda df: df["perc"].map(lambda x: x**z - 1)))
#     yyy = (zzz
#         .loc[:,["president","perc","party"]]
#         .assign(color = lambda df: df.party.map(lambda x: 'blue' if x=='Democrat' else 'red'))
#         .drop_duplicates()
#         .sort_values('perc'))
#     fig, ax = plt.subplots(1,1, figsize=(16,8));

#     ax.bar(yyy["president"],yyy["perc"], color=yyy['color']);
#     plt.setp(ax.get_xticklabels(), ha="right", rotation=45);
#     ax.set_title(f"Average Annual Percentage Change in\n{LABEL_MAP[x]} by President");
#     ax.set_ylabel(f"Average Annual Percentage Change in\n{LABEL_MAP[x]}");
#     ax.set_xlabel("President");
#     remove_borders(ax)
#     return fig
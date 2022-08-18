from datetime import datetime
import dash
import dash_table
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dash_table import FormatTemplate
import plotly.graph_objects as go
from dash.dash_table.Format import Format

from redshfit_connect import get_all_data_rs
from helper import npdt_to_str, get_default_row
from mc_model import predict_mc_model
from implied_price_dist import get_gradient, get_implied_price_distribution


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Get initial data from Redshift (variable names need to be in global scope)
futures_price = float(get_all_data_rs('futures_latest_o')['last_price'])
futures_data: pd.DataFrame = get_all_data_rs('futures_daily_d')

options_data: pd.DataFrame = get_all_data_rs('options_hourly_latest2_o')
options_data['strike'] = options_data.strike.astype(float)
options_data['best_bid_price'] = options_data.best_bid_price.astype(float)
options_data['best_ask_price'] = options_data.best_ask_price.astype(float)
options_data['last_price'] = options_data.last_price.astype(float)
options_data['volume'] = options_data.volume.astype(float)
options_data['open_interest'] = options_data['open_interest'].astype(float)
options_data['IV'] = options_data['ask_iv_'].astype(float)
options_data['delta'] = options_data['delta_'].astype(float)
options_data['gamma'] = options_data['gamma_'].astype(float)
options_data['vega'] = options_data['vega_'].astype(float)
options_data['theta'] = options_data['theta_'].astype(float)
options_data['rho'] = options_data['rho_'].astype(float)

display_columns = ['instrument_name', 'strike',
                'expiration_time', 'last_price', 'volume', 'open_interest', 'IV',
                'delta', 'gamma', 'vega', 'theta', 'rho']      
options_data_view = options_data[display_columns].sort_values(['strike'])



unique_expirations = options_data['expiration_time'].unique()
unique_expirations.sort()

default_row = get_default_row(options_data, unique_expirations[2])


strike_slider_step = 10000
strike_slider_min = int(max(0, (futures_price//strike_slider_step)
                        * strike_slider_step - 3*strike_slider_step))
strike_slider_max = int((futures_price//strike_slider_step)
                        * strike_slider_step + 2*strike_slider_step + 30000)


style_data_conditional = [
    {
        "if": {"state": "active"},
        "backgroundColor": "rgba(150, 180, 225, 0.2)",
        "border": "1px solid blue",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0, 116, 217, .03)",
        "border": "1px solid blue",
    },
]


calL_put = ['Call', 'Put']

money = FormatTemplate.money(2)
percentage = FormatTemplate.percentage(2)


columns_dt = [
    dict(id='instrument_name', name='instrument name'),
    dict(id='strike', name='strike', type='numeric', format=money),
    dict(id='last_price', name='last price', type='numeric', format=money),
    dict(id='volume', name='volume'),
    dict(id='open_interest', name='open interest'),
    dict(id='IV', name='IV', type='numeric', format=percentage),
    dict(id='delta', name='delta', type='numeric', format=Format(precision=2)),
    dict(id='gamma', name='gamma', type='numeric', format=Format(precision=2)),
    dict(id='vega', name='vega', type='numeric', format=Format(precision=2)),
    dict(id='theta', name='theta', type='numeric', format=Format(precision=2)),
    dict(id='rho', name='rho', type='numeric', format=Format(precision=2)),
]


app.layout = html.Div(

    children=[
        dcc.Interval(
            id='interval-component',
            interval=10*60*1000, # in milliseconds
            n_intervals=0
        ),
        html.H1(children='BTC Options Implied Price Distribution', className="header-title", style={
            'color':'black', 
            "padding": "2rem 1rem",}),
        
        # dcc.Markdown(children="Last refreshed data: ", id='card-body'),

        dcc.Markdown('''

            ### Notes        
            
            See below an approximated implied future price distribution of BTC price inferred from [Deribit](https://www.deribit.com/options/BTC) options data. Deribit options data is collected through the [API](https://docs.deribit.com/) every 10 minutes.

            The implied price distribution's densities are inferred by constructing butterflies at various strike prices and comparing the price of the butterfly to the the maximum payoff of the butterfly. Data points are smoothed using a gaussian filter before they are used to construct Splines using scipy.interpolate.CubicSpline.

            We also implement a monte carlo (MC) model with a control variate to calculate fair options prices (currently under development).

            Check out the [git repo](https://github.com/EricOstr/implied_btc_price_distribution) for more detail.


            #### Improvement Ideas:

            * Create a model forecasting the probability distribution of BTC price for given date - compare performance of this model to option market's prediction
            * Calculate option price with monte carlo using forecasted price distribution
            

            #### Improvement Ideas (Technical):

            * Migrate to AWS CDK for CI/CD
            * Separate from monolithic architecture to a micro service based one (dedicated service for MC simulation, volatility modelling etc)
            * Use C++ for faster MC simulation
            

            Creator of this website is [Eric Östring](https://www.linkedin.com/in/ericostring/)

            '''),
        html.Div([

            dcc.Loading(
                id="loading-0",
                type="default",
                children=[
                    dcc.Graph(id='price_dist'),
                    dcc.Markdown(id='price_dist_markdown')
                    ]
            )
        ]),



        # html.Div([

        #     dcc.Loading(
        #         id="loading-1",
        #         type="default",
        #         children=[
        #             dcc.Graph(id='mc_model'),
        #             dcc.Markdown(id='markdown')
        #             ]
        #     )
        # ]),
        # dbc.Alert(id='tbl_out'),

        # html.Div(className='row', style=dict(display='flex'), children=[
        #     html.Div(className='four columns', children=[
        #         dcc.Dropdown(
        #             id="expiration_date",
        #             options=[{"label": npdt_to_str(expr), "value": expr}
        #                      for expr in unique_expirations],
        #             value=unique_expirations[2],
        #             multi=False)
        #     ], style=dict(width='30%')),
        #     html.Div(className='four columns', children=[
        #         dcc.Dropdown(
        #             id="option_kind",
        #             options=[{"label": kind, "value": kind[:1]}
        #                      for kind in ['Call', 'Put']],
        #             value='C',
        #             multi=False)
        #     ], style=dict(width='30%')),
        #     html.Div(className='four columns', children=[
        #         dcc.RangeSlider(
        #             id='strike-slider',
        #             min=strike_slider_min,
        #             max=strike_slider_max,
        #             step=strike_slider_step,
        #             value=[strike_slider_min + 1*strike_slider_step,
        #                    strike_slider_max - 2*strike_slider_step],
        #             marks={i: "{:,}".format(i) for i in range(strike_slider_min, strike_slider_max+strike_slider_step, strike_slider_step)})
        #     ], style=dict(width='40%')),

        # ]
        # ),
        # dcc.Loading(
        #     id="loading-2",
        #     type="default",
        #     children=html.Div(
        #         html.Div(
        #             dash_table.DataTable(
        #                 id='options_table',
        #                 columns=columns_dt,
        #                 data=options_data_view.to_dict('records'),
        #                 style_data_conditional=style_data_conditional,
        #             )
        #         )
        #     )
        # ),




        
        html.Div(id='placeholder', style={'display':'none'})
    ],
    style={
        "margin-left": "2rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
)


@app.callback(
    Output('placeholder', 'children'),
    Input('interval-component', 'n_intervals'),
)
def refresh_data(n):
    '''
    Reconstruct global variables each time
    '''

    global futures_price
    global futures_data
    global options_data
    global unique_expirations

    # Get latest data from Redshift
    futures_price = float(get_all_data_rs('futures_latest_o')['last_price'])


    futures_data = get_all_data_rs('futures_daily_d')
    mask = ((datetime.now().timestamp() - futures_data.update_time.astype('datetime64[s]').astype('int')) / (60*60*24*7)) < 8 # Futures data for the past 2 months
    futures_data = futures_data[mask]
    futures_data = futures_data.sort_values(['update_time'])
    futures_data.reset_index(drop=True)    

    options_data = get_all_data_rs('options_hourly_latest2_o')
    options_data['strike'] = options_data.strike.astype(float)
    options_data['best_bid_price'] = options_data.best_bid_price.astype(float)
    options_data['best_ask_price'] = options_data.best_ask_price.astype(float)
    options_data['last_price'] = options_data.last_price.astype(float)
    options_data['volume'] = options_data.volume.astype(float)
    options_data['open_interest'] = options_data.open_interest.astype(float)
    options_data['IV'] = options_data['ask_iv_'].astype(float)
    options_data['delta'] = options_data['delta_'].astype(float)
    options_data['gamma'] = options_data['gamma_'].astype(float)
    options_data['vega'] = options_data['vega_'].astype(float)
    options_data['theta'] = options_data['theta_'].astype(float)
    options_data['rho'] = options_data['rho_'].astype(float)    
    

    unique_expirations = options_data['expiration_time'].unique()
    unique_expirations.sort()

    return n



@app.callback(
    Output('price_dist', 'figure'),
    Input('placeholder', 'children'),
)
def update_price_dist(n):

    global futures_price
    global futures_data
    global options_data

    # Calculate implied price distribution
    percentiles = np.arange(0.1,1,0.1)
    price_grad = get_implied_price_distribution(options_data, percentiles, futures_price)
    
    # Get historical data and connect to implied futures price distribution
    connect_time = futures_data.update_time.max()
    connect_price = futures_data[futures_data.update_time == connect_time]['last_price'].values[0]
    price_grad.loc[len(price_grad)] = [connect_time] + [connect_price for _ in percentiles]
    price_grad = price_grad.sort_values(['expiration_time'])
    price_grad = price_grad.reset_index(drop=True)

    max_color = 120 # the smaller the darker
    min_color = 210
    max_opacity = 0.7
    min_opacity = 0.5

    color_grad = get_gradient(max_color, min_color, len(percentiles))
    opacity_grad = get_gradient(max_opacity, min_opacity, len(percentiles))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=futures_data['update_time'], y=futures_data['last_price'], name="historical price", mode="lines", marker_color='rgba(60, 60, 60, .9)'))
    for pct, color, opacity in zip(percentiles, color_grad, opacity_grad):
        fig.add_trace(go.Scatter(x=price_grad["expiration_time"].values, y=price_grad[pct], name=f"{int(pct*100)}% percentile", mode="lines", marker_color=f'rgba({color}, {color}, {color}, {opacity})'))

    fig.update_layout(
        title="BTC Implied Price Distribution", xaxis_title="Date", yaxis_title="Price ($)"
    )

    return fig



@app.callback(
    Output('options_table', 'data'),
    Output('card-body', 'children'),
    
    Input('strike-slider', 'value'),
    Input('expiration_date', 'value'),
    Input('option_kind', 'value'),
    Input('placeholder', 'children'),
)
def filter_options_table(strike_slider_values, expr, kind, n):

    mask = (options_data['strike'] >= strike_slider_values[0]) \
        & (options_data['strike'] <= strike_slider_values[1]) \
        & options_data['expiration_time'].apply(lambda x: x.timestamp() == expr/1000000000) \
        & (options_data.apply(lambda x: x['instrument_name'][-1:] == kind, axis=1))

    display_columns = ['instrument_name', 'strike',
                    'expiration_time', 'last_price', 'volume', 'open_interest', 'IV',
                    'delta', 'gamma', 'vega', 'theta', 'rho']        
    
    last_refreshed = options_data['update_time'][0].strftime("%Y-%m-%d %H:%M:%S")

    options_data_view = options_data[mask][display_columns].sort_values(['strike'])

    # last_updated_text = 'Last Refreshed Data: ' + last_refreshed

    return options_data_view.to_dict('records'), 'Last Refreshed Data: ' + last_refreshed


@app.callback(
    Output('mc_model', 'figure'),
    Output("options_table", "style_data_conditional"),
    Output("markdown", "children"),
    Input('options_table', 'active_cell'),
    Input('options_table', 'derived_viewport_data'),
    Input('placeholder', 'children'),    
)
def update_mc_model(active, info, n):
    style = style_data_conditional.copy()
    if active:
        style.append(
            {
                "if": {"row_index": active["row"]},
                "backgroundColor": "rgba(150, 180, 225, 0.2)",
                "border": "1px solid blue",
            },
        )


    values_active = info[active['row']] if active is not None and active['row']<len(info) else default_row

    # Calculate MC model predicted price. Returns data for histogram
    data, C0, SE = predict_mc_model(
        flag=values_active['instrument_name'][-1:],
        S0=futures_price,
        K=values_active['strike'],
        vol=values_active['IV'],
        expr=values_active['expiration_time']
    )

    fig = px.histogram(data, 
                        histnorm='percent',
                        title='Simulated Options Price - Monte Carlo with Control Variate ',
                        labels={'value':"simulated price", 'percent':"density %"},
                        opacity=0.8,
                        range_x=[np.percentile(data, 2.5), np.percentile(data, 97.5)] if len(data) != 0 else None
                        )
    if len(data) != 0:

        fig.add_vline(C0, line_color='black')
        fig.add_vline(C0-2*SE, line_dash='dash', line_color="black")
        fig.add_vline(C0+2*SE, line_dash='dash', line_color="black")

        fig.add_vline(values_active['last_price'], line_color='red')

    
        fig.add_annotation(
                            dict(font=dict(color='black',size=12)),
                            x=np.percentile(data, 97.5) - 0.17*(np.percentile(data, 97.5)-np.percentile(data, 2.5)), 
                            y=26,
                            text=f"Exp. Discounted Payoff: ${round(C0,2)}",
                            showarrow=False,
                            )
        fig.add_annotation(
                            dict(font=dict(color='black',size=12)),
                            x=np.percentile(data, 97.5) - 0.17*(np.percentile(data, 97.5)-np.percentile(data, 2.5)), 
                            y=24,
                            text=f"Std. Error: ${round(SE,2)}",
                            showarrow=False,
                            )        
        fig.add_annotation(
                            dict(font=dict(color='red',size=12)),
                            x=np.percentile(data, 97.5) - 0.17*(np.percentile(data, 97.5)-np.percentile(data, 2.5)), 
                            y=22,
                            text=f"Last Price: ${round(values_active['last_price'],2)}",
                            showarrow=False,
                            )                                    

    markdown_text = f'''
                
                Expected Discounted Option Payoff:  ${round(C0,2)}\n
                Standard Error:                     ${round(SE,2)}\n
                \n
                \n
                \n
                '''

    return fig, style, markdown_text


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
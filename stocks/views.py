import base64
import datetime
import matplotlib
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from django.shortcuts import render
from django.http import JsonResponse
from django.utils.html import escape
matplotlib.use('Agg')
from io import BytesIO
import base64
from django.http import JsonResponse
import seaborn as sns


from matplotlib import dates
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, timedelta

# from stocks.utils import StockUtil
import json
import joblib
sns.set_style('whitegrid')

loaded_model = joblib.load('stocks/models/random_forest_model.pkl')
loaded_encoder = joblib.load('stocks/models/label_encoder.pkl')

import json
from django.shortcuts import HttpResponse


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


def ajax_test(request):
    if is_ajax(request=request):
        message = "This is ajax"
    else:
        message = "Not ajax"
    return HttpResponse(message)

def index(request):
    return render(request,"index.html")

def stock_form(request):
    return render(request,"chart.html")


def is_ajax(request):
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'


def trend_graph(request):
    if request.method == 'POST':
        # Get the stock name and time period from the submitted form
        stock_name = request.POST.get('stock_name', '')
        time_period = request.POST.get('time_period', '1mo')

        # Get stock data from yfinance
        stock_data = yf.download(stock_name, period=time_period)

        # Generate a graph for the closing price using plotly
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Closing Price'))
        fig1.update_layout(title_text='Closing Price for ' + escape(stock_name), xaxis_tickangle=-45)

        # Generate a graph for the moving average using plotly
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'].rolling(window=20).mean(), name='Moving Average'))
        fig2.update_layout(title_text='Moving Average for ' + escape(stock_name), xaxis_tickangle=-45)

        # Generate a graph for the volume using plotly
        fig3 = px.bar(stock_data, x=stock_data.index, y='Volume', title='Volume for ' + escape(stock_name))
        fig3.update_xaxes(tickangle=-45)

        # Encode the graphs to base64 format and store in variables
        buffer1 = BytesIO()
        fig1.write_image(buffer1, format='png')
        buffer1.seek(0)
        image1_png = buffer1.getvalue()
        buffer1.close()
        graph1 = base64.b64encode(image1_png).decode('utf-8')

        buffer2 = BytesIO()
        fig2.write_image(buffer2, format='png')
        buffer2.seek(0)
        image2_png = buffer2.getvalue()
        buffer2.close()
        graph2 = base64.b64encode(image2_png).decode('utf-8')

        buffer3 = BytesIO()
        fig3.write_image(buffer3, format='png')
        buffer3.seek(0)
        image3_png = buffer3.getvalue()
        buffer3.close()
        graph3 = base64.b64encode(image3_png).decode('utf-8')

        # Get the latest price of the stock
        latest_price = stock_data['Close'][-1]

        # Create a summary about the stock
        summary = {
            'name': stock_name,
            'price': latest_price,
            'time_period': time_period,
        }
        graph_data = {
            'graph1': graph1,
            'graph2': graph2,
            'graph3': graph3,
            'summary': summary,
        }

        # If the request was made with AJAX, return the graph data and summary as a JSON response
        if is_ajax(request=request):
            return JsonResponse(graph_data)

        # If the request was not made with AJAX, render the template with the graphs and summary embedded
        return render(request, 'trend_graph.html', {'graph1': graph1, 'graph2': graph2, 'graph3': graph3, 'summary': summary})

    # If the form hasn't been submitted yet, render the template for the form
    return render(request, 'stock_form.html')



def multiple_trend_graph(request):
    if request.method == 'POST':
        # Get the stock names and time period from the submitted form
        stock_name1 = request.POST.get('stock_name1', '')
        stock_name2 = request.POST.get('stock_name2', '')
        time_period = request.POST.get('time_period', '1mo')

        # Get stock data from yfinance for both stocks
        stock_data1 = yf.download(stock_name1, period=time_period)
        stock_data2 = yf.download(stock_name2, period=time_period)

        # Generate a graph for the first stock using matplotlib
        fig1_1 = go.Figure()
        fig1_1.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'], name='Closing Price',line=dict(color='rgb(107, 165, 124)')))
        fig1_1.update_layout(title_text='Closing Price for ' + escape(stock_name1), xaxis_tickangle=-45)
        
        fig1_1.update_layout(
        plot_bgcolor='rgb(225, 227, 227 )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig1_1.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph1_1 = base64.b64encode(image_png1).decode('utf-8')

        # Generate a graph for the moving average using plotly
        fig1_2 = go.Figure()
        fig1_2.add_trace(go.Scatter(x=stock_data1.index, y=stock_data1['Close'].rolling(window=20).mean(), name='Moving Average',line=dict(color='rgb(107, 165, 124)')))
        fig1_2.update_layout(title_text='Moving Average for ' + escape(stock_name1), xaxis_tickangle=-45)
        
        fig1_2.update_layout(
        plot_bgcolor='rgb(225, 227, 227   )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig1_2.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph1_2 = base64.b64encode(image_png1).decode('utf-8')
        

        
        # Generate a graph for the volume using plotly
        fig1_3 = px.bar(stock_data1, x=stock_data1.index, y='Volume', title='Volume for ' + escape(stock_name1),color_discrete_sequence=['rgb(107, 165, 124)'])
        fig1_3.update_xaxes(tickangle=-45)

        # Encode the first graph image to base64 format and store in a variable
        
        fig1_3.update_layout(
        plot_bgcolor='rgb(225, 227, 227 )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig1_3.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph1_3 = base64.b64encode(image_png1).decode('utf-8')

        # Generate a graph for the second stock using matplotlib
        fig2_1 = go.Figure()
        fig2_1.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'], name='Closing Price',line=dict(color='rgb(49, 187, 231)')))
        fig2_1.update_layout(title_text='Closing Price for ' + escape(stock_name2), xaxis_tickangle=-45)
        
        fig2_1.update_layout(
        plot_bgcolor='rgb(225, 227, 227   )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig2_1.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph2_1 = base64.b64encode(image_png1).decode('utf-8')
        
         #Generate a graph for the moving average using plotly
        fig2_2 = go.Figure()
        fig2_2.add_trace(go.Scatter(x=stock_data2.index, y=stock_data2['Close'].rolling(window=5).mean(), name='Moving Average',line=dict(color='rgb(49, 187, 231)')))
        fig2_2.update_layout(title_text='Moving Average for ' + escape(stock_name2), xaxis_tickangle=-45)
        
        fig2_2.update_layout(
        plot_bgcolor='rgb(225, 227, 227   )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig2_2.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph2_2 = base64.b64encode(image_png1).decode('utf-8')

        # Generate a graph for the volume using plotly
        fig2_3 = px.bar(stock_data2, x=stock_data2.index, y='Volume', title='Volume for ' + escape(stock_name2),color_discrete_sequence=['rgb(49, 187, 231) '])
        fig2_3.update_xaxes(tickangle=-45)


       
        fig2_3.update_layout(
        plot_bgcolor='rgb(225, 227, 227  )',  # Light gray
        paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        
        buffer1 = BytesIO()
        fig2_3.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph2_3 = base64.b64encode(image_png1).decode('utf-8')

        # Get the latest price of each stock
        latest_price1 = stock_data1['Close'][-1]
        latest_price2 = stock_data2['Close'][-1]

        # Create a summary about each stock
        summary1 = {
            'name': stock_name1,
            'price': latest_price1,
            'time_period': time_period,
        }
        summary2 = {
            'name': stock_name2,
            'price': latest_price2,
            'time_period': time_period,
        }

        graph_data = {
            'graph1_1': graph1_1,
            'graph1_2': graph1_2,
            'graph1_3': graph1_3,
            'graph2_1': graph2_1,
            'graph2_2': graph2_2,
            'graph2_3': graph2_3,
            'summary1': summary1,
            'summary2': summary2,
        }

        # If the request was made with AJAX, return the graph data and summary as a JSON response
        if is_ajax(request=request):
            return JsonResponse(graph_data)

        # If the request was not made with AJAX, render the template with the graphs and summaries embedded
        #return render(request, 'trend_graph.html', {'stock_name': stock_name, 'graph': graph, 'summary': summary})
    
    # If the form hasn't been submitted yet, render the template for the form
    return render(request, 'multiple_stock_form.html')



import plotly.graph_objs as go

def predict_stock(request):
    predictions = []
    graph = None
    
    if request.method == 'POST':
        stock = request.POST.get('stock_name')
        numofdays = request.POST.get('num_days')
        df = yf.download(stock, start='2023-04-24', end='2023-04-25')
        df = df.reset_index()
        for n in range(int(numofdays)):
            date_n_days_ago = datetime.now() + timedelta(days=n)
            df['DayOfWeek'] = date_n_days_ago.day
            df['Month'] = date_n_days_ago.month

            df['Symbol'] = loaded_encoder.fit_transform([stock])

            features = ['Open', 'DayOfWeek', 'Month', 'Symbol']
            target = 'Close'

            # Split the features and target variable for training and testing sets
            test_features = df[features]
            #print(test_features)
            pred = loaded_model.predict(test_features)
            predictions.append(pred[0])
            df['Open'] = pred[0]
            #print(pred)

        # Create a plotly graph for the predictions
        fig = go.Figure()
        x = list(range(len(predictions)))
        fig.add_trace(go.Scatter(x=x, y=df['Close'], name='Actual'))
        fig.add_trace(go.Scatter(x=x, y=predictions, name='Predicted'))
        fig.update_layout(title=f"Predicted Close Price for {stock} for Next {numofdays} Days",
                        xaxis_title="Number of Predictions",
                        yaxis_title="Close Price",
                        legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)'))

        fig.update_layout(
            plot_bgcolor='rgb(225, 227, 227   )',  # Light gray
            paper_bgcolor='rgb(242, 242, 242)',  # Light gray
        )
        
        buffer1 = BytesIO()
        fig.write_image(buffer1, format='png')
        buffer1.seek(0)
        image_png1 = buffer1.getvalue()
        buffer1.close()
        graph = base64.b64encode(image_png1).decode('utf-8')

    context = {'predictions': predictions, 'graph': graph}
    return render(request, 'predict_stock.html', context)
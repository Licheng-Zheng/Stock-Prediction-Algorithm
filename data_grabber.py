# important for making the API call to alpha vantage 
import requests

# The python file my_secrets contains the api code used as well as the list of stocks that I would like to get data for 
# The list consists of three stock tickers ["AAPL", "IBM", "BA"]
from my_secrets import api_key, stock_list

# Used to create the numpy array to store all the data within in a concise manner
import numpy as np

def do_stuff(stock_ticker):
    '''
    Takes the stock ticker and gets the data for everything
    '''

    # If the stock ticker is "All", return immediately as no specific stock data is needed
    if stock_ticker == "All":
        return

    # Construct the URL for the API call to Alpha Vantage
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_ticker}&outputsize=full&apikey={api_key}'

    # Make the API request
    r = requests.get(url)
 
    # Parse JSON response
    json_data = r.json()
    usable_data = json_data["Time Series (Daily)"]
    
    # Create a list with all the dates for which data has been collected
    all_keys = list(usable_data.keys())
    
    data_list  = []

    # Iterate over all the dates and extract the relevant data
    for key in all_keys:    
        data = usable_data[key]
        
        # Append the data for each date to the data_list
        data_list.append([
            float(data["1. open"]),
            float(data["2. high"]),
            float(data["3. low"]),
            float(data["4. close"]),
            int(data["5. volume"])
        ])

    # Convert the data_list to a numpy array
    array = np.array(data_list)

    print(array)

    # Save the numpy array to a file named after the stock ticker
    np.save(f"{stock_ticker}_data.npy", array)
    print(f"grabbed! {stock_ticker}")

def refresh_code():
    # I'm gonna do the stocks Apple (AAPL), IBM (IBM) and Boeing (BA), I could do more, but its 12:37 AM so I do not want to do more :D (might need to look at more stocks so I can get their stock information)

    # Executes the function do_stuff, which does the stuff (it refreshes the stuff) 
    for item in stock_list: 
        do_stuff(item)

refresh_code()


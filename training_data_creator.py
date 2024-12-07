import numpy as np 
from my_secrets import stock_list

def create(stock_ticker):
    all_information_list = []
    
    for ticker in stock_list: 
        try: 
            file_name = f"{ticker}_data.npy"

            # Loads in the file with all the stock information of a particular company, information to be uploaded to information list
            data = np.load(file_name)
            
            # Gets the shape of the data, mostly used to make sure there's nothing wrong with the data
            shape_of_data = data.shape
            print(shape_of_data)
            
            # Only takes date in which there are more than 7 days ahead and 30 days behind, gives it enough information to predict
            for actual_date in range(7, shape_of_data[0] - 31):

                # Information of the current data point (30 days behind and the number to predict)
                current_chunk = []

                # Stores the information of the future 7 days to compile the correct number
                correct_data_compiler = []
                
                # Gets the forward few days open and close information and makes it into a  list
                for future_information in range(1, 8):
                    correct_data_compiler.append(
                        data[actual_date - future_information][1]
                    )
                    correct_data_compiler.append(
                        data[actual_date - future_information][2]
                    )
                 
                # Average of the correct_data_compiler list is created
                number_to_append = sum(correct_data_compiler)/len(correct_data_compiler)
                
                # Gets all the information of the past 30 days and appends it to a list 
                for past_information in range(1, 31):
                    current_piece = data[actual_date + past_information].tolist()
                    for piece_of_information in current_piece: 
                        current_chunk.append(piece_of_information)
                        
                current_chunk.append(number_to_append)
                
                # All information added to a list which is made into a numpy array later on
                all_information_list.append(current_chunk)
                
        except FileNotFoundError: 
            print(f"File not created yet :( go to data_grabber and grab information for {ticker}")         
    

    array_of_information = np.array(all_information_list)
    print(array_of_information.shape)
    print(array_of_information[-1].shape)
    np.save(f"all_information.npy", array_of_information)
    print("List created!")
    
def create_dataset():
    # Tells the program which stocks you want to iterate over
    create("All")
    
create_dataset()

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Weather_Station:
    def __init__(self, data):
        self.df = data
        self.input = []
        self.label = []
        self.class_names = ['None', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Violent Rain']
   
   
    def graphics(self, column_1, column_2, xlabel, ylabel):
        plt.style.use("dark_background")
        plt.gcf().canvas.set_window_title('IoT Weather Station')
        plt.hist2d(self.df[column_1], self.df[column_2])
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(xlabel)
        plt.show()
    
    
    def data_visualization(self):
        self.graphics('WD', '1h_RF', 'Wind Direction (deg)', 'One-Hour Rainfall (mm)')
        self.graphics('Av_WS', '1h_RF', 'Average Wind Speed (m/s)', 'One-Hour Rainfall (mm)')
        self.graphics('Mx_WS', '1h_RF', 'Maximum Wind Speed (m/s)', 'One-Hour Rainfall (mm)')
        self.graphics('24h_RF', '1h_RF', '24-Hour Rainfall (mm)', 'One-Hour Rainfall (mm)')
        self.graphics('Tem', '1h_RF', 'Temperature (C)', 'One-Hour Rainfall (mm)')
        self.graphics('Hum', '1h_RF', 'Humidity (%)', 'One-Hour Rainfall (mm)')
        self.graphics('b_PR', '1h_RF', 'Barometric Pressure (hPA)', 'One-Hour Rainfall (mm)')    
    
    
    def scale_data(self):
        # Wind Direction and Speed:
        wv = self.df.pop('Av_WS')
        max_wv = self.df.pop('Mx_WS')
        # Convert to radians.
        wd_rad = self.df.pop('WD')*np.pi / 180
        # Calculate the wind x and y components.
        self.df['scaled_WX'] = wv*np.cos(wd_rad)
        self.df['scaled_WY'] = wv*np.sin(wd_rad)
        # Calculate the max wind x and y components.
        self.df['scaled_max_WX'] = max_wv*np.cos(wd_rad)
        self.df['scaled_max_WY'] = max_wv*np.sin(wd_rad)
        # Temperature:
        tem = self.df.pop('Tem')
        self.df['scaled_Tem'] = tem / 25
        # Humidity:
        hum = self.df.pop('Hum')
        self.df['scaled_Hum'] = hum / 70 
        # Barometric Pressure:
        bPR = self.df.pop('b_PR')
        self.df["scaled_bPR"] = bPR / 1013     
        # 24 Hour Rainfall (Approx.)
        rain_24 = self.df.pop('24h_RF')
        self.df['scaled_24h_RF'] = rain_24 / 24
   

    def create_input_and_label(self):
        n = len(self.df)

        for i in range(n):
            self.input.append(np.array([self.df['scaled_WX'][i], self.df['scaled_WY'][i], self.df['scaled_max_WX'][i], self.df['scaled_max_WY'][i], self.df['scaled_Tem'][i], self.df['scaled_Hum'][i], self.df['scaled_bPR'][i]]))
        self.input = np.asarray(self.input)
   
        for i in range(n):
            _class = 0
      
            approx_RF_rate = (self.df['1h_RF'][i] + self.df['scaled_24h_RF'][i]) * 100
           
            if approx_RF_rate == 0:
                _class = 0
            elif approx_RF_rate < 2.5:
                _class = 1
            elif 2.5 < approx_RF_rate and approx_RF_rate < 7.6:
                _class = 2
            elif 7.6 < approx_RF_rate and approx_RF_rate < 50:
                _class = 3
            else:
                _class = 4
            self.label.append(_class)
        self.label = np.asarray(self.label)
   
   
    def split_data(self):
        n = len(self.df)
        # (60%, 40%) - (training, test)
        self.train_input = self.input[0:int(n*0.6)]
        self.test_input = self.input[int(n*0.6):]
        self.train_label = self.label[0:int(n*0.6)]
        self.test_label = self.label[int(n*0.6):]
    
    
    def build_and_train_model(self):
       
        self.model = keras.Sequential([
            keras.Input(shape=(7,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),   
            keras.layers.Dense(5, activation='softmax')
        ])
        # Compile:
        self.model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        # Train:
        self.model.fit(self.train_input, self.train_label, epochs=19)
        # Test the accuracy:
        print("\n\nModel Evaluation:")
        test_loss, test_acc = self.model.evaluate(self.test_input, self.test_label) 
        print("Evaluated Accuracy: ", test_acc)
  

    def make_prediction(self, pre_array):
        print("\n\nModel Predictions:\n")
        prediction = self.model.predict(pre_array)
        for i in range(len(pre_array)):
            print("Prediction => ", self.class_names[np.argmax(prediction[i])])
    
    
    def save_model(self):
        self.model.save("/Users/rohitmeshram/Downloads/py-project/model.h5")

 
    def Neural_Network(self, save):
        self.scale_data()
        self.create_input_and_label()
        self.split_data()
        self.build_and_train_model()
        if save == True:
            self.save_model()
        
    
csv_path = "/Users/rohitmeshram/Downloads/py-project/dataset.csv"
df = pd.read_csv(csv_path)

# Define a new class object named 'station':
station = Weather_Station(df)

# Visualize data:
station.data_visualization()

# Artificial Neural Network (ANN):
station.Neural_Network(True)

# Inputs for making predictions:
prediction_array = np.array([
    [0, 0, 0.31819805, 0.31819805, 0.6988, 0.81498571, 0.99349753],
    [0, -0, 0, -0, 0.8444, 1, 0.96835143],
    [0, 0, 0.45, 0, 0.87577, 0.95857143, 1.00128332],
    [-0, -0, -0, -0, 0.8224, 1.05714286, 0.99279368]
])

# Prediction Results:
station.make_prediction(prediction_array)

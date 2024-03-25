
# Stock Future Predictor

The Stock Future Detector is a project focused on predicting future stock prices using deep learning techniques. This repository contains the code for training a Long Short-Term Memory (LSTM) neural network on historical stock market data to forecast future price trends. By leveraging historical stock price data and technical indicators, the model aims to provide insights into potential future price movements, enabling users to make informed investment decisions.

See it live at:- [https://stock-future-predicter--ankitkumarsinha.streamlit.app/](https://stock-future-predicter-ankitkumarsinha.streamlit.app/)
## Tech stack used
- Data Processing: NumPy, Pandas
- Data Visualization: Matplotlib, Seaborn
- Data Retrieval: Pandas Datareader, yfinance
- Deep Learning Framework: TensorFlow
- Deep Learning Library: Keras (integrated with TensorFlow)
- Machine Learning Model: LSTM (Long Short-Term Memory)
- Normalization: Sklearn
- Model Evaluation: Mean Squared Error, Mean Absolute Error
- Optimization Algorithm: Adam
- IDE (Integrated Development Environment): VS Code


## Model Architecture
- Deep Learning Model Architecture: Long Short-Term Memory (LSTM)

- Layers:

The model comprises multiple LSTM layers, including units 50, 60, 80, and 120, facilitating sequential data processing and feature extraction from the historical BTC-USD price data.

Each LSTM layer is accompanied by a dropout layer with varying dropout rates (0.2, 0.3, 0.4, and 0.5), aiding in regularization and preventing overfitting.

The final dense layer is utilized for predicting the output.
- Activation Function: Rectified Linear Unit (ReLU)

- Loss Function: Mean Squared Error (MSE)

- Optimizer: Adam, a popular optimization algorithm known for its efficiency and adaptability in training deep learning models.

This comprehensive LSTM-based architecture is well-suited for time-series analysis, enabling the model to capture complex patterns and dependencies within the BTC-USD price data. The implementation of ReLU activation functions, dropout layers, and the Adam optimizer ensures the model's stability, effective learning, and accurate forecasting capabilities.
## Training the model
The model was trained for 50 epochs and these are the plots:
- Model loss
![Model loss](https://github.com/ankitsinha4791/Stock-Future-Predicter/assets/97782545/f406d0ec-7086-4e0f-be4f-baba494f87a8)
- Model mean absolute error
![Model mean absolute error](https://github.com/ankitsinha4791/Stock-Future-Predicter/assets/97782545/bdc6d4d5-5241-4377-886a-f8c5d09e64f4)


## Results
Now, the best model (the one with the best lowest mean absoute error) predicts the stock:
- Loss :- 0.0021
- Absolute error :- 0.0320
Original vs Predicted
![Original vs Predicted](https://github.com/ankitsinha4791/Stock-Future-Predicter/assets/97782545/2d095c61-f781-43d9-a101-0ffa714050e3)

## Streamlit App
I have developed a user-friendly and interactive Streamlit app that leverages advanced deep learning models to predict stock prices. The app is designed to offer real-time insights and forecasts for better decision-making in the dynamic stock market environment. Users can seamlessly input their desired stock ticker and receive comprehensive analyses and visualizations of historical data trends, moving averages, and future price predictions, all through a streamlined and intuitive interface.

Link:- [https://stock-future-predicter--ankitkumarsinha.streamlit.app/](https://stock-future-predicter-ankitkumarsinha.streamlit.app/)
## Help & Support

Thank you for taking the time to explore our project! We appreciate your interest and support. If you have any questions, feedback, or encounter issues, please don't hesitate to reach out. We value your input and contributions.

Feel free to visit my website

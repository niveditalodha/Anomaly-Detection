# Project 4: Anomaly Detection in Time Evolving Networks

The algorithm in the code is the implementation of the paper NetSimile - A scalable approach to size independent network similarity

Python Version: 3.8.2

Python Libraries Required:
1. Numpy 1.18.4 - Install using "pip install numpy==1.18.4"
2. Networkx 2.8.7 - Install using "pip install networkx==2.8.7"
3. Scipy 1.4.1 - Install using "pip install scipy==1.4.1"
4. Matplotlib 3.4.3 - Install using "pip install matplotlib==3.4.3"

Run the algorithm:
1. Navigate into the terminal to the project folder where anomaly.py is present
2. The dataset folder consists of 4 different datasets. Choose one graph for example "voices"
3. Now run the command python anomaly.py graph. In this example, run python anomaly.py voices

Result of the algorithm:
1. After execution, we can see the threshold value and anomalies detected list in the terminal
2. We can see the plot file in the result folder with the name graph_time_series_plot.png. In this case, voices_time_series_plot.png
3. We can see the time series text file in the same result folder with the name graph_time_series.txt. In this case, voices_time_series.txt

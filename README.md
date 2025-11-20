# NVIDIA Stock
![Nvidia](/image.jpg)

## Procedures
- Import Libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Acquistion and Loading
    - Data acquired from Yahoo Finance API
- Data Preprocessing
    - Check for missing values
    - Check for duplicated rows
- Feature Engineering
    - Define feature matrix (X) and target vector (y)
- Data Splitting
    - Split the data into training (80%) and testing (20%) sets
- Data Scaling
    - Standardize features by removing the mean and scaling to unit variance
- Model Definition
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Decison Tree Regression
    - Random Forest Regression
- Hyperparameter Tuning
- Model Comparison and Evaluation
    - R2 Score
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Tuning Params
    - Best Performing Model: Linear Regression (R2 Score: 1.000000)
- Post-Training Visualization
    ![output](/output.png)
- New Prediction Input Function

## Process
![Screenshot (222)](/Screenshot%20(222).png)
![Screenshot (223)](/Screenshot%20(223).png)
![Screenshot (224)](/Screenshot%20(224).png)
![Screenshot (225)](/Screenshot%20(225).png)
![Screenshot (226)](/Screenshot%20(226).png)
![Screenshot (227)](/Screenshot%20(227).png)
![Screenshot (228)](/Screenshot%20(228).png)



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/nvidia-stock.git
cd customer-personality
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
nvidia-stock/
│
├── model.ipynb  
|── model.py    
├── requirements.txt 
├── image.jpg   
|── nvidia_stock_data.csv
|── output.png
├── Screenshot (222).png
├── Screenshot (223).png
├── Screenshot (224).png
├── Screenshot (225).png
├── Screenshot (226).png
├── Screenshot (227).png
├── Screenshot (228).png
├── LICENSE
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CONTRIBUTING.md
└── README.md          

```
## Tools and Dependencies
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

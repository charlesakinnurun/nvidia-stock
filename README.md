# NVIDIA Stock
![Nvidia](/image.jpg)

## Procedures
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
|── marketing_campaign.csv  
├── requirements.txt 
├── image.jpg   
|── output.png
|── LICENSE
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
## Contributing
Contributions are welcome! If you’d like to suggest improvements — e.g., new modelling algorithms, additional feature engineering, or better documentation — please open an Issue or submit a Pull Request.
Please ensure your additions are accompanied by clear documentation and, where relevant, updated evaluation results.

## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE)
 file for details.
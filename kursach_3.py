import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

def read_data(path):
    # Reads the file with the given delimiter ","
    data = pd.read_csv(path, sep=';')
    return data

def check_data(data):
    # Checks the data
    print("\n Data check")
    print(data.head(100))
    print(data.info())
    print(data.describe())
    # Count na
    print("\n Count na")
    print(data.isna().sum())


def linear_regression(X_train, X_test, y_train, y_test):
    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scatter_plot(y_test, y_pred, 'Linear Regression')
    return mse

def random_forest_regression(X_train, X_test, y_train, y_test):
    # Random Forest regression
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scatter_plot(y_test, y_pred, 'Random Forest Regression')
    return mse

def gradient_boosting_regression(X_train, X_test, y_train, y_test):
    # Gradient Boosting regression
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    scatter_plot(y_test, y_pred, 'Gradient Boosting Regression')
    return mse

def visualize_regression_results(mse_scores):
    models = ['Linear Regression', 'Random Forest Regression', 'Gradient Boosting Regression']
    
    plt.bar(models, mse_scores)
    plt.xlabel('Regression Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of Regression Models')
    plt.show()

def scatter_plot(y_test, y_pred, name):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Actual vs Predicted Score for '+name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()






if __name__ == "__main__":
    # read file
    print("\n Reading data imdb250")
    data_path_imdb = 'data/fact_imdb250.csv'
    dataset_imdb = read_data(data_path_imdb)
    check_data(dataset_imdb)

    print("\n Reading data movies")
    data_path_movies = 'data/dim_movies.csv'
    dataset_movies = read_data(data_path_movies)
    check_data(dataset_movies)

    # Merge datasets on "id" and "movie_id"
    merged_dataset = pd.merge(dataset_movies, dataset_imdb, left_on='id', right_on='title_id')

    data_path_rottentomatoes = 'data/fact_rotten_tomatoes.csv'
    dataset_rottentomatoes = read_data(data_path_rottentomatoes)
    check_data(dataset_rottentomatoes)
     # Check merged dataset
    print("\n Merged dataset")
    check_data(merged_dataset)

    # Drop unnecessary columns
    merged_dataset = merged_dataset.drop(columns=['id_y','certificate', 'synopsis', 'aspect_ratio', 'director', 'crew', 'producer', 'writer', 'sound_mix', 'collection', 'production_co'])

    #Merge on "id" and title_id
    merged_dataset_two = pd.merge(merged_dataset, dataset_rottentomatoes, left_on='title_id', right_on='title_id')

    merged_dataset_two = merged_dataset_two.drop(columns=['title_id','id','type','consensus','year_id','time_id'])
    merged_dataset_two = merged_dataset_two.replace(',', '', regex=True)


    check_data(merged_dataset_two)
    print(merged_dataset_two.head(10))

    merged_dataset_two.loc[merged_dataset_two['box_office'] <= 500, 'box_office'] = 0
    merged_dataset_two.loc[merged_dataset_two['box_office'] > 500, 'box_office'] = 1

    # Select features and target variable
    X = merged_dataset_two[['genre_id', 'language_id', 'box_office', 'runtime','year','release_date_id','rating_id','critic_score','people_score','total_reviews','total_ratings']]
    y = merged_dataset_two['score']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse_scores = [
    linear_regression(X_train, X_test, y_train, y_test),
    random_forest_regression(X_train, X_test, y_train, y_test),
    gradient_boosting_regression(X_train, X_test, y_train, y_test)
    ]

    # Visualize the results
    #This code will generate a bar plot comparing the mean squared error (MSE) for each regression model. The lower the MSE, the better the model's performance.
    visualize_regression_results(mse_scores)








    



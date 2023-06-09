import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

def read_data(path):
    # Reads the file with the given delimiter ","
    data = pd.read_csv(path, sep=',')
    return data

def read_data_semi(path):
    # Reads the file with the given delimiter ";"
    data = pd.read_csv(path, sep=';')
    return data

def check_data(data):
    # Checks the data
    print("\n Data check")
    print(data.head(10))
    print(data.info())
    print(data.describe())
    # Count na
    print("\n Count na")
    print(data.isna().sum())


def transform_runtime_to_minutes(runtime):
    if pd.isnull(runtime):
        return 108
    else:
        runtime = str(runtime)
        if 'h' in runtime:
            hours, minutes = runtime.split('h')
            hours = int(hours.strip())
            if len(minutes) !=0:
                minutes = int(minutes.strip().replace('m', ''))
            else:
                minutes = 0
            total_minutes = hours * 60 + minutes
        else:
            minutes = int(runtime.strip().replace('m', ''))
            total_minutes = minutes
        return total_minutes
    
def change_genre(genre, dataset_genre):
    genre = str(genre)
    if ',' in genre:
        type = genre.split(',')[0]
        id = dataset_genre[dataset_genre['name'] == type]['id'].values
    else:
        id = dataset_genre[dataset_genre['name'] == genre]['id'].values
    if id is None or len(id) == 0:
        return 24
    if len(id)>0:
        return id[0]
    return id
    
def change_rating(rating, dataset_rating):
    #if rating is nan then return 6
    if pd.isnull(rating):
        return 6
    rating = str(rating)
    if ' ' in rating:
        type = rating.split(' ')[0]
        #find in column name in dataset_genre 'type' and return id
        id = dataset_rating[dataset_rating['rating_name'] == type]['idrating'].values
    else:
        id = dataset_rating[dataset_rating['rating_name'] == rating]['idrating'].values
    if id is None or len(id) == 0:
        return 6
    if len(id)>0:
        return id[0]
    else:
        return id
    
def change_total_ratings(total):
    if pd.isnull(total):
        return total
    else:
        total = str(total)
        if 'Fewer' in total:
            total = total = total.split('Fewer than ')[1]
        if '+' in total:
            total = total.replace('+', '')
        if ' ' in total:
            total = total.split(' ')[0]
        total = total.replace(',', '')
        return int(total)
    
def change_original_language(language, dataset_language):
    if pd.isnull(language):
        return 32
    else:
        language = str(language)
        if ' ' in language:
            language = language.split(' ')[0]
        id = dataset_language[dataset_language['name'] == language]['id'].values
        if id is None or len(id) == 0:
            return 32
        if len(id)>0:
            return int(id[0])
        return id

def linear_regression(X_train, X_test, y_train, y_test):
    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #score with r-squared
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression R-squared: ", r2)
    scatter_plot(y_test, y_pred, 'Linear Regression')
    return r2

def random_forest_regression(X_train, X_test, y_train, y_test):
    # Random Forest regression
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest Regression R-squared: ", r2)
    scatter_plot(y_test, y_pred, 'Random Forest Regression')
    return r2

def lasso_regression(X_train, X_test, y_train, y_test):
    # Lasso regression
    model = Lasso()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Lasso Regression R-squared: ", r2)
    scatter_plot(y_test, y_pred, 'Lasso Regression')
    return r2

def visualize_regression_results(mse_scores):
    models = ['Linear Regression', 'Random Forest Regression','Lasso Regression']
    plt.bar(models, mse_scores)
    plt.xlabel('Regression Models')
    plt.ylabel('R-square')
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
    print("\n Reading data dataset_rotten_tomatoes")
    data_path_rotten_tomatoes  = 'data/rotten_tomatoes_top_movies.csv'
    dataset_rotten_tomatoes = read_data(data_path_rotten_tomatoes)
    data_path_genre = 'data/dim_genre.csv'
    dataset_genre = read_data_semi(data_path_genre)
    data_path_rating = 'data/dim_rating.csv'
    dataset_rating = read_data_semi(data_path_rating)
    data_path_language = 'data/dim_language.csv'
    dataset_language = read_data_semi(data_path_language)


    dataset_rotten_tomatoes=dataset_rotten_tomatoes.drop(columns=['title','type','box_office_(gross_usa)','view_the_collection','aspect_ratio','sound_mix','release_date_(theaters)','production_co','consensus','synopsis','crew','link','director','producer','writer','release_date_(streaming)'])




    dataset_rotten_tomatoes['genre'] = dataset_rotten_tomatoes['genre'].apply(change_genre, dataset_genre=dataset_genre)
    dataset_rotten_tomatoes['rating'] = dataset_rotten_tomatoes['rating'].apply(change_rating, dataset_rating=dataset_rating)
    dataset_rotten_tomatoes['original_language'] = dataset_rotten_tomatoes['original_language'].apply(change_original_language, dataset_language=dataset_language)


    dataset_rotten_tomatoes['total_ratings'] = dataset_rotten_tomatoes['total_ratings'].apply(change_total_ratings)
    dataset_rotten_tomatoes['runtime'] = dataset_rotten_tomatoes['runtime'].apply(transform_runtime_to_minutes)

    # change na 
    dataset_rotten_tomatoes['runtime'].fillna(dataset_rotten_tomatoes['runtime'].mean(), inplace=True)
    dataset_rotten_tomatoes['people_score'].fillna(dataset_rotten_tomatoes['people_score'].mean(), inplace=True)
    dataset_rotten_tomatoes['people_score'] = dataset_rotten_tomatoes['people_score'].astype(int)
    check_data(dataset_rotten_tomatoes)


    #     # Select features and target variable
    X = dataset_rotten_tomatoes[['year', 'people_score', 'total_reviews', 'total_ratings',  'runtime', 'genre', 'rating', 'original_language']]
    y = dataset_rotten_tomatoes['critic_score']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scores = [
    linear_regression(X_train, X_test, y_train, y_test),
    random_forest_regression(X_train, X_test, y_train, y_test),
    lasso_regression(X_train, X_test, y_train, y_test)
    ]

    # Visualize the results
    #This code will generate a bar plot comparing the mean squared error (MSE) for each regression model. The lower the MSE, the better the model's performance.
    visualize_regression_results(scores)



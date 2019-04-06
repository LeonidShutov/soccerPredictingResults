import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
import main

warnings.simplefilter("ignore")


def train_classifier(clf, dm_reduction, x_train, y_train, cv_sets, params, scorer, jobs, use_grid_search=True,
                     best_components=None, best_params=None):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()

    # Check if grid search should be applied
    if use_grid_search:

        # Define pipeline of dm reduction and classifier
        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)

        # Grid search over pipeline and return best classifier
        grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
        grid_obj.fit(x_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:

        # Use best components that are known without grid search
        estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)
        best_pipe = pipeline.fit(x_train, y_train)

    end = time()

    # Print the results
    print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    # Return best pipe
    return best_pipe


def predict_labels(clf, best_pipe, features, target):
    """ Makes predictions using a fit classifier based on scorer. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target.values, y_pred)


def train_calibrate_predict(clf, dm_reduction, x_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                            params, scorer, jobs,
                            use_grid_search=True, **kwargs):
    """ Train and predict using a classifer based on scorer. """

    # Indicate the classifier and the training set size
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))

    # Train the classifier
    best_pipe = train_classifier(clf, dm_reduction, x_train, y_train, cv_sets, params, scorer, jobs)

    # Calibrate classifier
    print("Calibrating probabilities of classifier...")
    start = time()
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv='prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    # Print the results of prediction for both training and testing
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         predict_labels(clf, best_pipe, x_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__,
                                                     predict_labels(clf, best_pipe, X_test, y_test)))

    # Return classifier, dm reduction, and label predictions for train and test set
    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, x_train, y_train), predict_labels(
        clf, best_pipe, X_test, y_test)


def convert_odds_to_prob(match_odds):
    """ Converts bookkeeper odds to probabilities. """

    # Define variables
    match_id = match_odds.loc[:, 'match_api_id']
    bookkeeper = match_odds.loc[:, 'bookkeeper']
    win_odd = match_odds.loc[:, 'Win']
    draw_odd = match_odds.loc[:, 'Draw']
    loss_odd = match_odds.loc[:, 'Defeat']

    # Converts odds to prob
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd

    total_prob = win_prob + draw_prob + loss_prob

    probs = pd.DataFrame()

    # Define output format and scale probs by sum over all probs
    probs.loc[:, 'match_api_id'] = match_id
    probs.loc[:, 'bookkeeper'] = bookkeeper
    probs.loc[:, 'Win'] = win_prob / total_prob
    probs.loc[:, 'Draw'] = draw_prob / total_prob
    probs.loc[:, 'Defeat'] = loss_prob / total_prob

    # Return probs and meta data
    return probs


def get_bookkeeper_data(matches, bookkeepers, horizontal=True):
    """ Aggregates bookkeeper data for all matches and bookkeepers. """

    bk_data = pd.DataFrame()

    # Loop through bookkeepers
    for bookkeeper in bookkeepers:

        # Find columns containing data of bookkeeper
        temp_data = matches.loc[:, (matches.columns.str.contains(bookkeeper))]
        temp_data.loc[:, 'bookkeeper'] = str(bookkeeper)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

        # Rename odds columns and convert to numeric
        cols = temp_data.columns.values
        cols[:3] = ['Win', 'Draw', 'Defeat']
        temp_data.columns = cols
        temp_data.loc[:, 'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:, 'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:, 'Defeat'] = pd.to_numeric(temp_data['Defeat'])

        # Check if data should be aggregated horizontally
        if horizontal:

            # Convert data to probs
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis=1, inplace=True)
            temp_data.drop('bookkeeper', axis=1, inplace=True)

            # Rename columns with bookkeeper names
            win_name = bookkeeper + "_" + "Win"
            draw_name = bookkeeper + "_" + "Draw"
            defeat_name = bookkeeper + "_" + "Defeat"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]

            # Aggregate data
            bk_data = pd.concat([bk_data, temp_data], axis=1)
        else:
            # Aggregate vertically
            bk_data = bk_data.append(temp_data, ignore_index=True)

    # If horizontal add match api id to data
    if horizontal:
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

    # Return bookkeeper data
    return bk_data


def find_best_classifier(classifiers, dm_reductions, scorer, x_t, y_t, x_c, y_c, x_v, y_v, cv_sets, params, jobs):
    """ Tune all classifier and dimensionality reduction combinations to find best classifier. """

    # Initialize result storage
    clfs_return = []
    dm_reduce_return = []
    train_scores = []
    test_scores = []

    # Loop through dimensionality reductions
    for dm in dm_reductions:

        # Loop through classifiers
        for clf in clfs:
            # Grid search, calibrate, and test the classifier
            clf, dm_reduce, train_score, test_score = train_calibrate_predict(clf=clf, dm_reduction=dm, x_train=x_t,
                                                                              y_train=y_t,
                                                                              X_calibrate=x_c, y_calibrate=y_c,
                                                                              X_test=x_v, y_test=y_v, cv_sets=cv_sets,
                                                                              params=params[clf], scorer=scorer,
                                                                              jobs=jobs, use_grid_search=True)

            # Append the result to storage
            clfs_return.append(clf)
            dm_reduce_return.append(dm_reduce)
            train_scores.append(train_score)
            test_scores.append(test_score)

    # Return storage
    return clfs_return, dm_reduce_return, train_scores, test_scores


def run():
    global clfs
    start = time()
    # Fetching data
    # Connecting to database
    path = "/Users/shiz/soft/database.sqlite"
    conn = sqlite3.connect(path)
    # Defining the number of jobs to be run in parallel during grid search
    n_jobs = 2  # Insert number of parallel jobs here
    # Fetching required data tables
    player_data = pd.read_sql("SELECT * FROM Player;", conn)
    player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    team_data = pd.read_sql("SELECT * FROM Team;", conn)
    match_data = pd.read_sql("SELECT * FROM Match;", conn)
    # Reduce match data to fulfill run time requirements
    rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    match_data.dropna(subset=rows, inplace=True)
    match_data = match_data.tail(1500)
    # Generating features, exploring the data, and preparing data for model training
    # Generating or retrieving already existant FIFA data
    fifa_data = main.get_fifa_data(match_data, player_stats_data, data_exists=False)
    # Creating features and labels based on data provided
    bk_cols_selected = ['B365', 'BW']
    feables = main.create_feables(match_data, fifa_data, bk_cols_selected, get_overall=True)
    inputs = feables.drop('match_api_id', axis=1)
    # Exploring the data and creating visualizations
    labels = inputs.loc[:, 'label']
    features = inputs.drop('label', axis=1)
    features.head(5)
    # Splitting the data into Train, Calibrate, and Test data sets
    x_train_calibrate, x_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=42,
                                                                            stratify=labels)
    x_train, x_calibrate, y_train, y_calibrate = train_test_split(x_train_calibrate, y_train_calibrate, test_size=0.3,
                                                                  random_state=42,
                                                                  stratify=y_train_calibrate)
    # Creating cross validation data splits
    cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
    cv_sets.get_n_splits(x_train, y_train)
    # Initializing all models and parameters
    # Initializing classifiers
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
    ab_clf = AdaBoostClassifier(n_estimators=200, random_state=2)
    gnb_clf = GaussianNB()
    knn_clf = KNeighborsClassifier()
    log_clf = linear_model.LogisticRegression(multi_class="ovr", solver="sag", class_weight='balanced')
    clfs = [rf_clf, ab_clf, gnb_clf, knn_clf, log_clf]
    # Specifying scorer and parameters for grid search
    feature_len = features.shape[1]
    scorer = make_scorer(accuracy_score)
    parameters_rf = {'clf__max_features': ['auto', 'log2'],
                     'dm_reduce__n_components': np.arange(5, feature_len, 5)}
    parameters_ab = {'clf__learning_rate': np.linspace(0.5, 2, 5),
                     'dm_reduce__n_components': np.arange(5, feature_len, 5)}
    parameters_gnb = {'dm_reduce__n_components': np.arange(5, feature_len, 5)}
    parameters_knn = {'clf__n_neighbors': [3, 5, 10],
                      'dm_reduce__n_components': np.arange(5, feature_len, 5)}
    parameters_log = {'clf__C': np.logspace(1, 1000, 5),
                      'dm_reduce__n_components': np.arange(5, feature_len, 5)}
    parameters = {clfs[0]: parameters_rf,
                  clfs[1]: parameters_ab,
                  clfs[2]: parameters_gnb,
                  clfs[3]: parameters_knn,
                  clfs[4]: parameters_log}
    # Initializing dimensionality reductions
    pca = PCA()
    dm_reductions = [pca]
    # Training a baseline model and finding the best model composition using grid search
    # Train a simple GBC classifier as baseline model
    clf = log_clf
    clf.fit(x_train, y_train)
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         accuracy_score(y_train, clf.predict(x_train))))
    print(
        "Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, accuracy_score(y_test, clf.predict(x_test))))
    # Training all classifiers and comparing them
    clfs, dm_reductions, train_scores, test_scores = find_best_classifier(clfs, dm_reductions, scorer, x_train, y_train,
                                                                          x_calibrate, y_calibrate, x_test, y_test,
                                                                          cv_sets,
                                                                          parameters, n_jobs)


run()

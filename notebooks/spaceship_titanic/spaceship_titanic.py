import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import PowerTransformer

from catboost import CatBoostClassifier, Pool, CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")


data = '../data/spaceship-titanic/'

df = pd.read_csv(f'{data}train.csv')
df_test = pd.read_csv(f'{data}test.csv')

cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include='float64').columns.tolist()
bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
luxury_cols = ['Spa',  'VRDeck']


def feature_engineering(df):
    df_copy = df.copy()
    df_copy['group'] = df_copy.apply(lambda row: row['PassengerId'].split('_')[0], axis=1)
    df_copy['num_group'] = df_copy.apply(lambda row: row['PassengerId'].split('_')[1], axis=1)
    df_copy['deck'] = df_copy.apply(lambda row: row['Cabin'].split('/')[0] if pd.notna(row['Cabin']) else row['Cabin'], axis=1)
    df_copy['num_cabin'] = df_copy.apply(lambda row: row['Cabin'].split('/')[1] if pd.notna(row['Cabin']) else row['Cabin'],
                               axis=1)
    df_copy['side_cabin'] = df_copy.apply(lambda row: row['Cabin'].split('/')[2] if pd.notna(row['Cabin']) else row['Cabin'],
                                axis=1)
    df_copy['last_name'] = df_copy.apply(lambda row: row['Name'].split()[1] if pd.notna(row['Name']) else row['Name'], axis=1)
    df_copy['total_spending'] = df_copy.apply(lambda row: row[bill_cols].sum(), axis=1)
    df_copy['total_spending_luxuries'] = df_copy.apply(lambda row: row[luxury_cols].sum(), axis=1)

    temp = df_copy.groupby('group').size().reset_index(name='group_size')
    df_copy = df_copy.merge(temp, how='left', on='group')

    return df_copy


def second_feature_engineering(df):
    df_copy = df.copy()
    df_copy['solo'] = df_copy['group_size'] > 1
    df_copy['same_destination'] = df_copy.groupby('group')['Destination'].transform(lambda x: x.nunique() == 1)
    df_copy['group_has_vip'] = df_copy.groupby('group')['VIP'].transform(lambda x: x.any())
    df_copy['is_child'] = df_copy['Age'] < 12
    df_copy['multiple_last_names'] = df_copy.groupby('group')['last_name'].transform(lambda x: x.nunique())
    df_copy['spent_anything'] = df_copy[bill_cols].sum(axis=1) > 0
    df_copy['bool_Spa'] = df_copy['Spa'] > 0
    df_copy['bool_VRDeck'] = df_copy['VRDeck'] > 0
    df_copy['same_cryosleep'] = df_copy.groupby('group')['CryoSleep'].transform(lambda x: x.nunique() == 1)
    df_copy['avg_group_spending'] = df_copy.groupby('group')['total_spending'].transform('mean')
    return df_copy


def generic_missing_values(df, train=True):
    df_copy = df.copy()

    age_median = df_copy['Age'].median()
    df_copy['Age'] = df_copy['Age'].fillna(age_median)

    zero_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in zero_cols:
        df_copy[col] = df_copy[col].fillna(0)

    mode_cols = ['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'side_cabin']
    for col in mode_cols:
        mode_value = df_copy[col].mode()[0]
        df_copy[col] = df_copy[col].fillna(mode_value)

    df_copy['deck'] = df_copy.groupby('HomePlanet')['deck'].transform(
        lambda x: x.fillna(x.mode()[0]))

    df_copy = fill_num_cabin(df_copy)

    return df_copy


def fill_num_cabin(df):
    df_copy = df.copy()
    cols = ['group', 'deck', 'num_cabin', 'group_size', 'side_cabin', 'HomePlanet', 'Spa', 'VRDeck', 'FoodCourt']
    df_cabin = df_copy.dropna(subset=['deck', 'side_cabin', 'HomePlanet'])

    df_cabin_train = df_cabin[~df_cabin['num_cabin'].isna()]
    df_cabin_prediction = df_cabin[df_cabin['num_cabin'].isna()]

    model = CatBoostRegressor(cat_features=['deck', 'side_cabin', 'HomePlanet'])
    model.fit(df_cabin_train[cols].drop('num_cabin', axis=1), df_cabin_train['num_cabin'], verbose=False)

    df_cabin_prediction['num_cabin'] = model.predict(df_cabin_prediction[cols].drop('num_cabin', axis=1))

    df_copy['num_cabin'] = df_copy['num_cabin'].combine_first(df_cabin_prediction['num_cabin'])

    return df_copy


def clean_missing_values(df):
    df_copy = df.copy()

    # In CryoSleep people can not spend
    filtered_rows = (df_copy['CryoSleep'] == True) & (df_copy[bill_cols].isna().any(axis=1))
    df_copy.loc[filtered_rows, bill_cols] = df_copy.loc[filtered_rows, bill_cols].fillna(0.0)

    # If people spent they would not be in CryoSleep
    cryosleep_false = (df_copy['CryoSleep'].isna()) & (df_copy[bill_cols].gt(0).any(axis=1))
    df_copy.loc[cryosleep_false, 'CryoSleep'] = False

    # people who did not spend anything unlikely were VIP
    vip_false = (df_copy['CryoSleep'] == False) & (df_copy[bill_cols].eq(0).all(axis=1)) & (df_copy['VIP'].isna())
    df_copy.loc[vip_false, 'VIP'] = False

    # People younger 18 and people from Earth doesn't have VIP
    df_copy.loc[((df_copy['VIP'].isna()) & (df_copy['Age'] < 18)), 'VIP'] = False
    df_copy.loc[((df_copy['VIP'].isna()) & (df_copy['HomePlanet'] == 'Earth')), 'VIP'] = False

    # last name with the most frequent in a group
    group_last_name_count = df_copy[df_copy['group_size'] > 1].groupby(['group', 'last_name']).size().reset_index(name='count')
    most_frequent_last_name = group_last_name_count.loc[group_last_name_count.groupby('group')['count'].idxmax()]
    df_copy = df_copy.merge(most_frequent_last_name[['group', 'last_name']], on='group', how='left',
                  suffixes=('', '_most_frequent'))
    df_copy['last_name'] = df_copy['last_name'].fillna(df_copy['last_name_most_frequent'])
    df_copy.drop(columns=['last_name_most_frequent'], inplace=True)
    df_copy['last_name'] = df_copy['last_name'].fillna('Unknown')

    # People from the same group has the same HomePlanet
    df_copy['HomePlanet'] = df_copy.groupby('group')['HomePlanet'].transform(lambda x: x.ffill().bfill())

    # Deck and HonePlanet are connected
    df_copy.loc[(df_copy['deck'] == 'A') & (df_copy['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
    df_copy.loc[(df_copy['deck'] == 'B') & (df_copy['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
    df_copy.loc[(df_copy['deck'] == 'G') & (df_copy['HomePlanet'].isna()), 'HomePlanet'] = 'Earth'
    df_copy.loc[(df_copy['deck'] == 'C') & (df_copy['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
    df_copy.loc[(df_copy['deck'] == 'T') & (df_copy['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'

    # People from the same planet doesn't share last names
    last_name_planet = df_copy[df_copy['last_name'] != 'Unknown'].groupby(['last_name',
                                                    'HomePlanet']).size().reset_index()[['last_name', 'HomePlanet']]
    df_copy = df_copy.merge(last_name_planet, on='last_name', how='left', suffixes=('', '_from_names'))
    df_copy['HomePlanet'] = df_copy['HomePlanet'].fillna(df_copy['HomePlanet_from_names'])
    df_copy.drop(columns=['HomePlanet_from_names'], inplace=True)

    # People younger 13 can not buy anything
    df_copy.loc[(df_copy[bill_cols].isna().any(axis=1)) & (df_copy["Age"] < 13), bill_cols] = 0.0

    # people from the same group has the same side cabin
    df_copy['side_cabin'] = df_copy.groupby('group')['side_cabin'].transform(lambda x: x.ffill().bfill())

    # if in the group there
    deck_unique_group = df_copy[df_copy['group_size'] > 1].groupby('group')['deck'].nunique(dropna=True).reset_index()
    groups_same_desk = deck_unique_group[deck_unique_group['deck'] == 1]
    filled_deck = df_copy[df_copy['group'].isin(groups_same_desk['group'])].groupby('group')['deck'].transform(
        lambda x: x.ffill().bfill())
    df_copy['deck'] = df_copy['deck'].combine_first(filled_deck)

    df_copy = generic_missing_values(df_copy)

    return df_copy


def transforming(df):
    df_copy = df.copy()
    transformer = PowerTransformer(method='yeo-johnson')
    cols_transform = ['RoomService', 'FoodCourt', 'ShoppingMall', 'total_spending_luxuries']
    for col in cols_transform:
        df_copy[[col]] = transformer.fit_transform(df_copy[[col]])
        # df_copy[[col]] = np.log1p(df_copy[[col]])
    return df_copy


def preprocessing(df):
    df_copy = df.copy()
    df_copy = feature_engineering(df_copy)
    df_copy = clean_missing_values(df_copy)
    df_copy = transforming(df_copy)
    return df_copy


def modeling(df, train_cols, cat_features):
    df_copy = df.copy()
    # df_copy = df_copy.dropna()
    df_copy['Transported'] = df_copy.apply(lambda row: 0 if row['Transported'] == False else 1, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df_copy[train_cols], df_copy['Transported'], test_size=0.2,
                                                        random_state=42)

    train_data = Pool(X_train, y_train, cat_features=cat_features)
    test_data = Pool(X_test, y_test, cat_features=cat_features)
    model = CatBoostClassifier(verbose=True, random_state=42)
    model.fit(train_data)

    y_pred = model.predict(test_data)

    return y_test, y_pred, model


def evaluation(y_test, preds_class):
    print(classification_report(y_test, preds_class))
    print('\n')
    print(f1_score(y_test, preds_class))


def feature_importance(model):
    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_

    # Create a DataFrame for easy plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('CatBoost Feature Importance')
    plt.gca().invert_yaxis()  # To have the highest importance on top
    plt.show()


def result_test(df, train_cols, model, v):
    df_copy = df.copy()
    # df_copy = preprocessing(df_copy)
    preds_class = model.predict(df_copy[train_cols])
    df_copy['Transported'] = preds_class
    df_copy['Transported'] = df_copy['Transported'].apply(lambda x: True if x == 1 else False)
    df_copy[['PassengerId', 'Transported']].to_csv(f'../submissions/spaceship_titanic/submission_{v}.csv',
                                                   index=False)


if __name__ == '__main__':
    df_copy = pd.concat([df, df_test], ignore_index=True)

    # df_copy = feature_engineering(df_copy)
    # df_copy = clean_missing_values(df_copy)
    # # df_copy = transforming(df_copy)
    df_copy = preprocessing(df_copy)
    df_copy = second_feature_engineering(df_copy)
    train = df_copy[~df_copy['Transported'].isna()]
    # print(df_copy.drop(['Cabin', 'Name'], axis=1).isna().sum())

    train_cols = [col for col in df_copy.columns if
                  col not in ['PassengerId', 'Cabin', 'Name', 'Transported', 'same_cryosleep','avg_group_spending',
                              'num_group', 'total_spending', 'Spa', 'VRDeck', 'group', 'solo', 'same_destination','VIP']]
    cat = [cols for cols in cat_cols if cols in train_cols]

    cat_features = cat + ['deck', 'side_cabin', 'multiple_last_names', 'is_child',
                          'last_name', 'spent_anything', 'bool_VRDeck', 'bool_Spa', 'group_has_vip']

    y_test, y_pred, model = modeling(train, train_cols, cat_features)
    evaluation(y_test, y_pred)
    feature_importance(model)

    test = df_copy[df_copy['Transported'].isna()]
    result_test(test, train_cols, model, 6)

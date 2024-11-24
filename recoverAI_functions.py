import pandas as pd
from fuzzywuzzy import process
from openai import OpenAI
import pickle
from datetime import datetime


client = OpenAI()

food_names = pd.read_excel('data/2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.xlsx')
alt_names  = pd.read_excel('data/2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx')


files = [food_names, alt_names]
for file in files:
    # make first row column names and drop first row
    file.columns = file.iloc[0]
    file.drop(file.index[0], inplace=True)
    # set index to Food Code
    file.set_index('Food code', inplace=True)

scores  = pd.read_csv('data/food_scores.csv', index_col='Unnamed: 0')
tags  = pd.read_csv('data/food_tags.csv', index_col='Unnamed: 0')

entries = food_names.copy()
entries = entries.drop(columns=['Main food description', 'WWEIA Category number', 'WWEIA Category description'])

with open('new_data/users.pkl', 'rb') as f:
    users = pickle.load(f)


def reset_users():
    users = []
    with open('new_data/users.pkl', 'wb') as f:
        pickle.dump(users, f)

def search_database(ingredient, names_df, alternatives_df, tags_df, scores_df, double=False):
    if double:
        results = 30
    else:
        results = 10
    matches = process.extract(ingredient, names_df['Main food description'], limit=10*results)
    all_matches = {}
    for match in matches:
        food_code = match[2]
        all_matches[food_code] = match[1]

    alt_matches = process.extract(ingredient, alternatives_df['Additional food description'], limit=results)
    for match in alt_matches:
        food_code = match[2]
        if food_code in all_matches:
            all_matches[food_code] += match[1]
        else:
            all_matches[food_code] = match[1]
    
    tag_matches = process.extract(ingredient, tags_df['Tags'], limit=results)
    for match in tag_matches:
        food_code = match[2]
        if food_code in all_matches:
            all_matches[food_code] += match[1]
        else:
            all_matches[food_code] = match[1]
    
    for key in all_matches:
        all_matches[key] *= scores_df['Score'][key]**1.5
    
    # return top 5 matches
    l_version = [(key, all_matches[key]) for key in all_matches]
    l_version.sort(key=lambda x: x[1], reverse=True)
    food_names = [names_df.loc[i[0]]['Main food description'] for i in l_version[:results]]
    return food_names

def get_first_question(ingredient):
    return f"You are a food and nutrition expert. Please tell me the ingredients in: {ingredient} Reply with a list of items and weights in grams in the following format [food, weight] * [food 2, weight 2] . It is important you return only this list, make a good estimate even if you don't know exactly. You may get a 1 ingredient item, in this case you can return a list of length 1.  Give all weights as numbers. Seperate items in the list with a comma and not line breaks between items"


def get_second_question(food_lists):
    return f'based on your previous list, I have found potential matches in my food database for each ingredient. So here are n lists of 10 descriptions. Please return the index(starting from 0) of the most appropriate entry. So you should return a list of n integers. If none of my entries are any good instead return an "E" for that entry. Food lists : {food_lists}'

def get_third_question(replacement_search_vals):
    return f'We have added many more search options for the food you could not find. Please give the index of the best of these. Return only a single integer. new options : {replacement_search_vals}'

def make_list(ent):
    new_ent = ent.split(',')
    # remove all white space
    new_ent = [i.strip() for i in new_ent]

    new_ent[0] = new_ent[0][1:]
    new_ent[-1] = new_ent[-1][:-1]
    return new_ent

def make_database_entry(food_codes, weights, entries):
    # find each food code in entries database and add that row*(weight/100) to the final result row 
    # pandas dataframe format
    final_result = pd.Series(0, index=entries.columns)
    for i in range(len(food_codes)):
        food_code = food_codes[i]
        weight = weights[i]
        row = entries.loc[food_code]
        final_result += row*(weight/100)
    return list(final_result)


def get_foodcodes(entry, verbose=False):
    response1 =  client.chat.completions.create(
    model="gpt-4o-mini",  # Replace with the exact model name if different
    messages=[
        {"role": "user", "content": get_first_question(entry)},
    ]
)
    list_vals = response1.choices[0].message.content
    ents = list_vals.split('*')

    weights = [make_list(i)[1] for i in ents]
    weights = [float(i) for i in weights]
    names = [make_list(i)[0] for i in ents]

    if verbose:
        print(f'Ingredients: {names}, with weights: {weights}')

    food_lists = [search_database(ingredient, food_names, alt_names, tags, scores) for ingredient in names]

    response2 = client.chat.completions.create(
    model="gpt-4o-mini",  # Replace with the exact model name if different
    messages=[
        {"role": "user", "content": get_first_question(entry)},
        {"role": "assistant", "content": list_vals},
        {"role": "user", "content": get_second_question(food_lists)},
    ]
    )
    best_indexes_raw = make_list(response2.choices[0].message.content)
    best_indexes = []

    for position,i  in enumerate(best_indexes_raw):
        if i == 'E':
            replacement_search_vals = search_database(names[position], food_names, alt_names, tags, scores, double=True)
            food_lists[position] = replacement_search_vals
            response3 = client.chat.completions.create(
                model="gpt-4o-mini",  # Replace with the exact model name if different
                messages=[
                    {"role": "user", "content": get_first_question(entry)},
                    {"role": "assistant", "content": list_vals},
                    {"role": "user", "content": get_second_question(food_lists)},
                    {"role": "assistant", "content": response2.choices[0].message.content},
                    {"role": "user", "content": get_third_question(replacement_search_vals)},
                ]
                )
            best_indexes.append(int(response3.choices[0].message.content))
        else:
            best_indexes.append(int(i))

    best_foods = [food_lists[count][i] for count, i in enumerate(best_indexes)]
    food_codes = [food_names[food_names['Main food description'] == i].index[0] for i in best_foods]
    if verbose:
        print(f'Chosen foods: {best_foods}')
    return food_codes, weights

def get_row(entry, verbose=False):
    food_codes, weights = get_foodcodes(entry, verbose=verbose)
    row = make_database_entry(food_codes, weights, entries)
    return row

def new_food_entry(user, entry, verbose=False):
    new_row = get_row(entry, verbose=verbose)
    new_row = [datetime.now()] + new_row

    if user in users:
        user_data = pd.read_csv(f'new_data/{user}.csv')
    else:
        user_data = pd.DataFrame(columns=['date_time'] + list(entries.columns))
        users.append(user)
        with open('new_data/users.pkl', 'wb') as f:
            pickle.dump(users, f)
    
    # add new row to user_data
    user_data.loc[len(user_data)] = new_row
    user_data.to_csv(f'new_data/{user}.csv', index=False)
    
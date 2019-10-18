import operator
from scipy.spatial import distance
import pandas as pd
import numpy as np


# reading, cleaning and combining the data.
def get_csv():
    books = pd.read_csv('BX-Books.csv', sep=';',
                        error_bad_lines=False, encoding="latin-1", low_memory=False)
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
                     'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

    users = pd.read_csv('BX-Users.csv', sep=';',
                        error_bad_lines=False, encoding="latin-1", low_memory=False)
    users.columns = ['userID', 'Location', 'Age']

    # filtering the users which only located in usa or canada
    users = users[users['Location'].str.contains("usa|canada")]

    ratings = pd.read_csv('BXBookRatingsTrain1.csv', sep=';',
                          error_bad_lines=False, encoding="latin-1", low_memory=False)
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    combined_list = pd.merge(books, ratings, on='ISBN')
    us_canada_user_rate = pd.merge(users, combined_list, on='userID')

    columns = ['Location', 'Age', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

    us_canada_user_rate.drop(columns, axis=1, inplace=True)

    # I will only use explicit data
    us_canada_user_rate = us_canada_user_rate[us_canada_user_rate['bookRating'] > 0]

    # filtered the dataset with items that were rated at least 1 times and
    # users who have rated at least 1 items.
    filtered_data = us_canada_user_rate.groupby('ISBN').filter(lambda x: len(x) >= 1)
    data = filtered_data.groupby('userID').filter(lambda x: len(x) >= 1)

    # Storing the data into a dataframe using the pandas.
    # then convert it a pivot table then dictionary to easily use.
    user_rating_pivot = data.pivot_table(index='ISBN', columns='userID', values='bookRating', fill_value=0)
    _dict = user_rating_pivot.to_dict()

    return users, books, _dict, user_rating_pivot

# reads BXBookRatingsTest1.csv, merge the data with users&books then returns the test dataframe.
def test():

    test = pd.read_csv('BXBookRatingsTest1.csv', sep=';', error_bad_lines=False, encoding="latin-1", low_memory=False)
    test.columns = ['userID', 'ISBN', 'predict']
    test = pd.merge(users, test, on='userID')
    test = pd.merge(books, test, on='ISBN')
    test = test.dropna()
    test = test[test['predict'] > 0]
    return test



########### Weighted kNN using Pearson Correlation or Manhattan distance ###########

# Find the K nearest neighbors based on the Manhattan distance
def find_manhattan_distance(target_uid, isbn_id, ubr_dict):

    try:
        target_user_rates = ubr_dict[target_uid]
    except:
        #print("User yok " + str(target_uid))
        return 406

    sim_dict = {}

    try:
        for user_id, other_user_rates in ubr_dict.items():
            if user_id != target_uid and ubr_dict[user_id][isbn_id] != 0:
                sim_dict[user_id] = 1/manhattan(target_user_rates, other_user_rates)
    except:
        #print("Kitap yok " + str(isbn_id))
        return 409

    return sim_dict

# Predict the rate for the provided data and
# classifier implementing the weighted k-nn rate.
def predict_with_wknn(target_uid, isbn_id, ubr_dict, k):
    sim_dict = find_pearson_sim(target_uid, isbn_id, ubr_dict)
    if type(sim_dict) == int:
        return sim_dict
    sorted_list = sorted(sim_dict.items(), key=operator.itemgetter(1))
    sorted_list.reverse()

    if len(sorted_list) == 0:
        return find_book_avg(isbn_id)

    l = k
    part = 0
    dom = 0
    for user in sorted_list[:l]:
        other_uid = user[0]
        weight = user[1]
        rate = ubr_dict[other_uid][isbn_id]
        # print(other_uid, weight, rate)
        part += rate * weight
        dom += weight


    try:
        return part/dom
    except:
        return 0


# The distance between two points measured along axes at right angles.
# In a plane with p1 at (x1, y1) and p2 at (x2, y2), it is |x1 - x2| + |y1 - y2|.
def manhattan(target_user_rates, other_user_rates):
    a1 = list(target_user_rates.values())
    a2 = list(other_user_rates.values())
    return distance.cityblock(a1, a2)


########### kNN using pearson similarity ###########

# Find the K nearest neighbors based on the Pearson Correlation
def find_pearson_sim(target_uid, isbn_id, ubr_dict):

    try:
        target_user_rates = ubr_dict[target_uid]
    except:
        return 406

    sim_dict = {}

    try:
        for user_id, other_user_rates in ubr_dict.items():
            if user_id != target_uid and ubr_dict[user_id][isbn_id] != 0:
                sim_dict[user_id] = pearson(target_user_rates, other_user_rates)
    except:
        return 409

    return sim_dict

# Predict the rate for the provided data
def predict_with_knn(target_uid, isbn_id, ubr_dict, k):

    sim_dict = find_pearson_sim(target_uid, isbn_id, ubr_dict)
    if type(sim_dict) == int:
        return sim_dict
    sorted_list = sorted(sim_dict.items(), key=operator.itemgetter(1))
    sorted_list .reverse()

    if len(sorted_list ) == 0:
        return find_book_avg(isbn_id)

    l = k
    part = 0

    for user in sorted_list [:l]:
        other_uid = user[0]
        rate = ubr_dict[other_uid][isbn_id]
        part += rate

    return part/k

# The Pearson product-moment correlation coefficient is a measure of
# the strength of the linear relationship between two variables.
# If the relationship between the variables is not linear, then the
# correlation coefficient does not adequately represent the strength of
# the relationship between the variables.
def pearson(target_user_rates, other_user_rates):

    a1 = list(target_user_rates.values())
    a2 = list(other_user_rates.values())
    return np.corrcoef(a1, a2)[0][1]


# if there is no user as 'uid',
# it returns the average score given to that book.
def find_book_avg(isbn_id):
    book_list = user_rating_pivot.loc[isbn_id].tolist()
    book_votes = [vote if vote != 0 else np.nan for vote in book_list]
    book_avg = np.nanmean(np.array(book_votes))
    return book_avg

# if there is no book as 'isbn_id',
# it returns corresponding user's average rate.
def find_user_avg(uid):
    vote_list = _dict[uid].values()
    vote_list = [vote if vote != 0 else np.nan for vote in vote_list]
    vote_avg = np.nanmean(np.array(vote_list))
    return vote_avg


def run_test(test_data, _dict, k, w):
    uid_list = test_data['userID'].tolist()
    isbn_list = test_data['ISBN'].tolist()
    rate_list = test_data['predict'].tolist()

    n = len(uid_list)

    part = 0
    c = 0

    for i in range(n):
        uid = uid_list[i]
        isbn_id = isbn_list[i]
        actual = rate_list[i]

        if w:
            predicted = predict_with_wknn(uid, isbn_id, _dict, k)
        else:
            predicted = predict_with_knn(uid, isbn_id, _dict, k)

        if predicted == 406:
            # there is no user in train set
            if isbn_id in list(_dict.values())[0]:
                predicted = find_book_avg(isbn_id)
            else:
                #print("Veri yok {} {}".format(uid, isbn_id))
                continue
        elif predicted == 409:
            # there is no book in train set
            if uid in _dict.keys():
                predicted = find_user_avg(uid)
            else:
                #print("Veri yok {} {}".format(uid, isbn_id))
                continue

        if predicted > 10: predicted = 10
        elif predicted < 0: predicted = 0
        c += 1
        part += abs(actual - predicted)

    result = part / c
    return result


users, books, _dict, user_rating_pivot = get_csv()
test = test()

mae_wknn_list = []
mae_knn_list = []

for i in range(1, 14):
    wknn = run_test(test, _dict, i, True)
    knn = run_test(test, _dict, i, False)

    mae_wknn_list.append(wknn)
    mae_knn_list.append(knn)

    print("k = " + str(i) + "W-kNN: " + str(run_test(test, _dict, i, True)))
    print("k = " + str(i) + "kNN: " + str(run_test(test, _dict, i, False)))
    print("")

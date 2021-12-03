import sys
from ast import parse

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import simplejson
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr


def sorting(df):
    actor_dic = {}
    actor_list = []
    for item in df:

        item = simplejson.loads(item)
        for j in item:
            if j["name"] in actor_dic.keys():
                actor_dic[j['name']] = actor_dic[j['name']] + 1
            else:
                actor_dic[j['name']] = 1

    actor_dic_sort = sorted(actor_dic.items(), key=lambda d: d[1],reverse=True)
    for i in actor_dic_sort:
         actor_list.append(i[0])
    return actor_list


def transfer(df):
    list = []
    item = simplejson.loads(df)
    for i in range(len(item)):
        list.append(item[i]["name"])
    return len(list)

def transfer_cast(df,name):
    item = simplejson.loads(df)
    for i in range(len(item)):
        if name==item[i]["name"]:
            return 1
    return 0

def transfer_crew(df,name):
    editor_list = []
    producer_list = []
    director_list = []
    writer_list = []
    item = simplejson.loads(df)
    for i in range(len(item)):
        if item[i]["job"]=='Editor':
            editor_list.append(item[i]['name'])

        elif item[i]["job"]=='Producer':
            producer_list.append(item[i]['name'])
        elif item[i]["job"]=='Director':
            director_list.append(item[i]['name'])
        if item[i]["department"]=='Art':
            writer_list.append(item[i]['name'])
    if name == 'Producer':
        return len(producer_list)
    elif name == 'Editor':
        return len(editor_list)
    elif name == 'Director':
        return len(director_list)
    elif name == 'Art':
        return len(writer_list)

def transfer_director(df):
    director_list = []
    item = simplejson.loads(df)
    for i in range(len(item)):

        if item[i]["job"]=='Director':
            director_list.append(item[i]['name'])

    return director_list


def crew_sort(df):
    actor_dic = {}
    actor_list = []
    for item in df:
        for j in item:
            if j in actor_dic.keys():
                actor_dic[j] = actor_dic[j] + 1
            else:
                actor_dic[j] = 1

    actor_dic_sort = sorted(actor_dic.items(), key=lambda d: d[1], reverse=True)
    for i in actor_dic_sort:
        actor_list.append(i[0])
    return actor_list

def cleansing1(df):
    train = pd.DataFrame()
    #df_dic = eval(df['cast'])

    popular_cast=sorting(df['cast'])
    popular_cast = popular_cast[0:14]
    for item in popular_cast:
        actor=item
        train[actor]=df['cast'].apply(transfer_cast,args=(actor,))

    train['Editor'] = df['crew'].apply(transfer_crew, args=('Editor',))#.02
    train['Producer'] = df['crew'].apply(transfer_crew, args=('Producer',))#.01

    num = []
    for item in df['crew']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['crew_num'] = pd.Series(num)
    num = []

    for item in df['cast']:
        content = simplejson.loads(item)
        num.append(len(content))

    train['cast_num'] = pd.Series(num)

    director = df['crew'].apply(transfer_director)
    all_director = crew_sort(director)
    popular_dir = all_director[0:15]
    for item in popular_dir:
        edi = []
        for j in director:
            if item in j:
                edi.append(1)
            else:
                edi.append(0)
        train[item] = pd.Series(edi)

    train['budget'] = df['budget'].apply(lambda x:x//10000000)#postive
    all_genres = sorting(df['genres'])
    all_genres = all_genres[0:5]
    for item in all_genres:
        train[item] = df['genres'].apply(transfer_cast,args=(item,))
    num = []
    for item in df['genres']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['genres_num'] = pd.Series(num)
    #train['genres'] = df['genres'].apply(transfer)
    popular_key = sorting(df['keywords'])
    popular_key = popular_key[0:8]
    for item in popular_key:
        train[item] = df['keywords'].apply(transfer_cast,args=(item,))
    #all_ori_lan = sorting(df['original_language'])
    num = []
    for item in df['keywords']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['key_num'] = pd.Series(num)
    ori_lan = []
    for item in df['original_language']:
        if item =='en':
            ori_lan.append(1)
        else:
            ori_lan.append(0)
    train["original_en"] = pd.Series(ori_lan)

    is_homepage = []
    for item in df['homepage']:
        if item is None:
            is_homepage.append(0)
        else:
            is_homepage.append(1)
    train['homepage'] = pd.Series(is_homepage)


    all_product_com = sorting(df['production_companies'])
    popular_com = all_product_com[0:8]
    for item in popular_com:
        train[item] = df['production_companies'].apply(transfer_cast,args=(item,))
    num = []
    for item in df['production_companies']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['com_num'] = pd.Series(num)

    all_product_cou = sorting(df['production_countries'])
    popular_cou = all_product_cou[0:6]
    for item in popular_cou:
        train[item] = df['production_countries'].apply(transfer_cast, args=(item,))

    df['runtime'] = df['runtime'].fillna(df['runtime'].mode()[0])
    train['runtime'] = df['runtime'].apply(lambda x:x/100)

    #train['spoken_languages'] = df['spoken_languages'].apply(transfer)
    popular_spoken = sorting(df['spoken_languages'])
    popular_spoken = popular_spoken[0:3]
    for item in popular_spoken:
        train[item] = df['spoken_languages'].apply(transfer_cast, args=(item,))

    season = []
    for item in df['release_date']:
        y = int(item[5:7])
        # season
        if y > 9:z = 4
        elif y > 6:z = 3
        elif y > 3:z = 2
        else:z = 1
        season.append(z)
    train['month'] = pd.Series(season)

    day = []
    for item in df['release_date']:
        y = int(item[8:10])
        if y>15:z =1
        else:z=0
        day.append(z)
    train['day'] = pd.Series(day)

    return train

def cleansing2(df):
    train = pd.DataFrame()

    train['Producer'] = df['crew'].apply(transfer_crew, args=('Producer',))
    train['Director'] = df['crew'].apply(transfer_crew, args=('Director',))

    num = []
    for item in df['crew']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['crew_num'] = pd.Series(num)



    all_genres = sorting(df['genres'])
    all_genres = all_genres[0:5]
    for item in all_genres:
        train[item] = df['genres'].apply(transfer_cast, args=(item,))
    num = []
    for item in df['genres']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['genres_num'] = pd.Series(num)


    train['budget'] = df['budget'].apply(lambda x:x//1000000)#postive
    df['runtime'] = df['runtime'].fillna(df['runtime'].mode()[0])
    train['runtime'] = df['runtime']

    num = []
    for item in df['production_companies']:
        content = simplejson.loads(item)
        num.append(len(content))
    train['com_num'] = pd.Series(num)

    train['US'] = df['production_countries'].apply(transfer_cast, args=('United States of America',))
    train['GB'] = df['production_countries'].apply(transfer_cast, args=('United Kingdom',))

    popular_spoken = sorting(df['spoken_languages'])
    popular_spoken = popular_spoken[0:8]
    for item in popular_spoken:
        train[item] = df['spoken_languages'].apply(transfer_cast, args=(item,))

    centre = []
    for item in df['release_date']:
        y = int(item[0:1])
        centre.append(y)
    train['year'] = pd.Series(centre)

    season=[]
    for item in df['release_date']:
        y = int(item[5:7])
        #season
        if y>9:z =4
        elif y>6:z = 3
        elif y > 3: z= 2
        else: z=1
        season.append(z)
    train['month']= pd.Series(season)

    return train



if __name__ == '__main__':


    source = pd.read_csv(sys.argv[1])
    test_source =pd.read_csv(sys.argv[2])

    train_x = cleansing1(source)
    train_y = source['revenue']
    test_x = cleansing1(test_source)
    test_y = test_source['revenue']
    test_y = np.array(test_y)


    #linear_model = linear_model.LinearRegression().fit(X=train_x,y=train_y)
    #test_y_2 = linear_model.predict(test_x)
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(X=train_x,y=train_y)
    test_y_2 = regressor.predict(test_x)
    df1 = pd.DataFrame()
    df1['zid'] = ['z5243684']
    df1['MSR'] = np.around([mean_squared_error(test_y, test_y_2)], 2)
    df1['correlation'] = np.around(np.corrcoef(test_y,test_y_2)[0,1], 2)
    df1.set_index('zid', inplace=True)
    df1.to_csv('revenue.PART1.summary.csv')

    df2 = pd.DataFrame(index=list(range(0,len(list(test_y)))))
    df2['movie_id'] = test_source[['movie_id']]
    df2['predicted_revenue'] = np.around(test_y_2)
    df2.set_index('movie_id', inplace=True)
    df2.to_csv('revenue.PART1.output.csv')

    #part2

    train_x = cleansing2(source)
    train_y = source['rating']
    test_x = cleansing2(test_source)
    test_y = test_source['rating']
    test_y = np.array(test_y)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X=train_x,y=train_y)
    test_y_2 = knn_model.predict(test_x)

    df3 = pd.DataFrame()
    #zid,average_precision,average_recall,accuracy
    df3['zid'] = ['z5243683']
    df3['average_precision'] =  np.around([precision_score(test_y, test_y_2, average='macro').mean()], 2)
    df3['average_recall'] = np.around([recall_score(test_y, test_y_2, average='macro').mean()], 2)
    df3['accuracy'] = np.around([accuracy_score(test_y, test_y_2)], 2)
    df3.to_csv('rating.PART2.summary.csv')

    df4 = pd.DataFrame(index=list(range(0,len(list(test_y)))))
    df4['movie_id'] = test_source[['movie_id']]
    df4['predicted_rating'] = np.around(test_y_2)
    df4.set_index('movie_id', inplace=True)
    df4.to_csv('rating.PART2.output.csv')






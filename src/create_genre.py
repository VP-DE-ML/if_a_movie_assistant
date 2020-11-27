
import pandas as pd
import json



df_genre_file = pd.read_excel(r'../input/Utterances.xlsx', sheet_name='genres_parts')
#print(df_genre_file)
list_genre_parts = df_genre_file.values.tolist()
#print(list_genre_parts)

df_unique_genres = pd.read_excel(r'../input/Utterances.xlsx', sheet_name='unique_genres')
#print(df_unique_genres)
list_unique_genres = df_unique_genres.values.tolist()
#print(list_unique_genres)

action = "retrieve_movie_by_genre"
final_utterance = []
short_list = []
my_NaN = float("NaN")
n = "n "
movie = 'movie'
film = 'film'
picture = 'picture'
for genre, genre1 in list_unique_genres:
    genre = str(genre).lower()
    genre1 = str(genre1).lower()
    for part1, part3 in list_genre_parts:
        part1 = str(part1).lower()
        part3 = str(part3).lower()
        if (genre[0] in ['a','e','i','o','u','h'] and part1[-1] == "a"):
            n = "n "
        else:
            n = " "
        if pd.isnull(part3) or part3 == "nan":
            print(str(part1) + " " + str(genre).strip('[]\''))
            value_movie = str(part1) + " " + genre1
        else:
            print(str(part1) + n + str(genre).strip('[]\'') + str(part3))
            value_movie = str(part1) + n + genre1 + str(part3)
        short_list = [genre, value_movie, "", action]
        final_utterance.append(short_list)

        value_film = value_movie.replace(movie, film)
        short_list = [genre, value_film, "", action]
        final_utterance.append(short_list)

        value_picture = value_movie.replace(movie, picture)
        short_list = [genre, value_picture, "", action]
        final_utterance.append(short_list)

value_list = []
previous_genre = ""
object_dict = {}
big_list = []
for genre, value, target, action in final_utterance:
    if previous_genre == "":
        previous_genre = genre
        genre1, value1, target1, action1 = genre, value, target, action
    if previous_genre == genre:
        value_list.append(value)
    else:
        object_dict = {
            "intent": genre1,
            "userinputs": value_list,
            "responses": target1,
            "action": action1
        }
        big_list.append(object_dict)
        value_list = []
        previous_genre = genre
        genre1, value1, target1, action1 = genre, value, target, action

final_obj = {
"intents": big_list
}


with open('../output/output.json', 'w') as outfile:
    json.dump(final_obj, outfile, sort_keys=False, indent=4)

print('Debuguer')
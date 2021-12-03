# data-movie
revenue of movies and the rating of a movie
##run: python3 main.py training.csv validation.csv
###revenue
As for how I evaluate the performance, it depends on the lower MSR and the higher correlation. Actually, I change several models when learning the revenue. What impressed me the most was LinearRegression model, although it’s easy to improve the correlation. The problem is the output exist negative value, which is impossible in the movie revenue. As for the feature selection, I list all features at first. The model can’t learn the string, so I change all the feature to the integres.

Popular cast: choose the most popular casts in the training data, because popular actors will increase the revenue of movies.
The number of casts:  The greater the number of actors, the greater the investment in the movie.
The number of editor & the number of producers:  the greater number of editor or producer may improve the revenue.
The number of casts:  The greater the number of crews, the greater the investment in the movie.
Popular director: choose the most famous directors.
Budget: The cost of the movie affects the quality of the movie.
The genres of movies: some popular genres may attract more custmers.
The popular keywords: some popular keywords may attract more custmers.
Successful production company & the number of companies : these production companies will have better capabilities to produce high-grossing movies.
Production country: this means movie distribution in more countries.
Spoken language: the wide distribution in the word
The release date(month, day): different release time will get different revenue.
###rating
As for how I evaluate the performance, I mainly judge based on accuracy and also consider the precision and recall.
In the rating, I find that the cast has little inference. 
On the contract, these features are important to learn. 
the producer and director 
the number of crews
the genres 
the number of production company
production countries(US, GB)
budget
spoken languages (popular languages)
runtime
release dates(year and month)

the problem:
there are different kinds of data to clean, so I need to use several methods.
Most of data when reading from csv is string, which is not easy to use. I use ‘simplejson’ to change the struct. When I want to use the same method for data processing, I change back by ‘str()’.



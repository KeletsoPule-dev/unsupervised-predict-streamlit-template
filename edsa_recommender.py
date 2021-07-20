"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import codecs
#Visuals
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.express as px

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
movies = pd.read_csv('resources/data/movies.csv')
train = pd.read_csv('resources/data/ratings.csv')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')

# Merging the train and the movies
df_merge1 = train.merge(movies, on = 'movieId')
from datetime import datetime
# Convert timestamp to year column representing the year the rating was made on merged dataframe
df_merge1['rating_year'] = df_merge1['timestamp'].apply(lambda timestamp: datetime.fromtimestamp(timestamp).year)
df_merge1.drop('timestamp', axis=1, inplace=True)

# -------------- Create a Figure that shows us that shows us how the Ratigs are distriuted. ----------------#
# Get the data
data = df_merge1['rating'].value_counts().sort_index(ascending=False)

ratings_df = pd.DataFrame()
ratings_df['Mean_Rating'] = df_merge1.groupby('title')['rating'].mean().values
ratings_df['Num_Ratings'] = df_merge1.groupby('title')['rating'].count().values

genre_df = pd.DataFrame(df_merge1['genres'].str.split('|').tolist(), index=df_merge1['movieId']).stack()
genre_df = genre_df.reset_index([0, 'movieId'])
genre_df.columns = ['movieId', 'Genre']

def make_bar_chart(dataset, attribute, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if sort_index == False:
        xs = dataset[attribute].value_counts().index
        ys = dataset[attribute].value_counts().values
    else:
        xs = dataset[attribute].value_counts().sort_index().index
        ys = dataset[attribute].value_counts().sort_index().values


    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.bar(x=xs, height=ys, color=bar_color, edgecolor=edge_color, linewidth=2)
    plt.xticks(rotation=45)

 # Merging the merge data earlier on with the df_imbd
df_merge3 = df_merge1.merge(df_imdb, on = "movieId" )

num_ratings = pd.DataFrame(df_merge3.groupby('movieId').count()['rating']).reset_index()
df_merge3 = pd.merge(left=df_merge3, right=num_ratings, on='movieId')
df_merge3.rename(columns={'rating_x': 'rating', 'rating_y': 'numRatings'}, inplace=True)

# pre_process the budget column

# remove commas
df_merge3['budget'] = df_merge3['budget'].str.replace(',', '')
# remove currency signs like "$" and "GBP"
df_merge3['budget'] = df_merge3['budget'].str.extract('(\d+)', expand=False)
#convert the feature into a float
df_merge3['budget'] = df_merge3['budget'].astype(float)
#remove nan values and replacing with 0
df_merge3['budget'] = df_merge3['budget'].replace(np.nan,0)
#convert the feature into an integer
df_merge3['budget'] = df_merge3['budget'].astype(int)

df_merge3['release_year'] = df_merge3.title.str.extract('(\(\d\d\d\d\))', expand=False)
df_merge3['release_year'] = df_merge3.release_year.str.extract('(\d\d\d\d)', expand=False)

data_1= df_merge3.drop_duplicates('movieId')

# Movies published by year:

years = []

for title in df_merge3['title']:
    year_subset = title[-5:-1]
    try: years.append(int(year_subset))
    except: years.append(9999)

df_merge3['moviePubYear'] = years
print('The Number of Movies Published each year:',len(df_merge3[df_merge3['moviePubYear'] == 9999]))

def make_histogram(dataset, attribute, bins=25, bar_color='#3498db', edge_color='#2980b9', title='Title', xlab='X', ylab='Y', sort_index=False):
    if attribute == 'moviePubYear':
        dataset = dataset[dataset['moviePubYear'] != 9999]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, pad=20)
    ax.set_xlabel(xlab, fontsize=16, labelpad=20)
    #ax.set_yticklabels([yticklabels(item, 'M') for item in ax.get_yticks()])
    ax.set_ylabel(ylab, fontsize=16, labelpad=20)

    plt.hist(dataset[attribute], bins=bins, color=bar_color, ec=edge_color, linewidth=2)

    plt.xticks(rotation=45)


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Movie Data Analysis", "Box Office Trailers", "Solution Overview", "About Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Movie Data Analysis":
        st.title("Exploratory Data Analysis")
        st.write("### Data analysis of Movie Data")
        st.markdown("What would you like to know about the movies that you are watching?"
                    " Here is some analytics that can help you make a choice about what to watch OR"
                    " help you win at your next movie trivia.")
        st.markdown(" ")
        st.image("resources/moviegif.gif")
        st.markdown("### Make a selection on the sidebar to explore the world of film.")
        if st.sidebar.checkbox("Insights on Ratings"):
            st.markdown("### What would you like to know?")
            if st.checkbox("How are the ratings distributed?"):
                f = px.histogram(df_merge1["rating"], x="rating", nbins=10, title="The Distribution of the Movie Ratings")
                f.update_xaxes(title="Ratings")
                f.update_yaxes(title="Number of Movies per rating")
                st.plotly_chart(f)
                st.markdown("According to a study, anything higher than about a mid-3 is considered enviably high,"
                            " and anything higher than a mid-3.5 is incredibly rare.Only a handful of 4s are given"
                            " every year; 4.5s are reserved for best-of-the-decade or even best-of-the-genre "
                            "material; The rating 5 is very rare.")
                st.markdown("We observe a large amount of rating of 4 which shows that a large number of viewers in"
                            " our dataset were substantially satisfied with the movies they watched.")
                st.markdown("We observe that ratings below 3 are rare in this case.")
                st.markdown(" ")


            if st.checkbox("How does the average movie rating sentiment change as more people rate movies?"):
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title('Average movie rating sentiment change as more people rate movies', fontsize=24, pad=20)
                ax.set_xlabel('Rating', fontsize=16, labelpad=20)
                ax.set_ylabel('Number of Ratings', fontsize=16, labelpad=20)
                plt.scatter(ratings_df['Mean_Rating'], ratings_df['Num_Ratings'], alpha=0.5, color='green')
                st.pyplot(fig)
                st.markdown("The more a movie gets more ratings it’s average ratings tends to increase."
                            " This is because the more people that watch the movie the more ratings it gets and "
                            "the more people talk about it and share it the more people like it, the movie is "
                            "hyped. Also, if a lot of people are watching a movie, chances are that they like it"
                            " and so they will rate it highly and share it with their friends and family.")
                st.markdown(" ")

            if st.checkbox("How many users rated movies over the years?"):
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                ax1 = df_merge1.groupby('rating_year')['rating'].count().plot(kind='bar', title='Ratings by year')
                st.write(fig)
                st.markdown("This shows that people tend to give lower ratings in recent years, "
                            "this may be because of the availability of a variety of movies in recent years, "
                            "making users stricter when it comes to reviewing movies. However we can not ignore "
                            "the possibility that our data is bias, maybe our data has more ratings for the more "
                            "recent years causing them to average out to 3.5 and it has less data for the earlier "
                            "years - year 2000 going back which means only a few ratings affect the average.")



        if st.sidebar.checkbox("Insights on Genres"):
            st.markdown("### What would you like to see?")
            if st.checkbox("What are the different movie genres on offer?"):
                st.image('resources/Wordcloud_of_the_movie_genres.png')
                st.markdown("In the early days of cinema, genres were much more uniform and defined. "
                            "Just as they were in literature and other forms of art and entertainment, "
                            "people would go to the theater to watch a war film, a musical, or a comedy. "
                            "The basic genres were well defined and included some of the following:")
                st.markdown('* ##### Action')
                st.markdown('* ##### Comedy')
                st.markdown('* ##### Drama')
                st.markdown('* ##### Fantasy')
                st.markdown('* ##### Horror')
                st.markdown('* ##### Mystery')
                st.markdown('* ##### Romance')
                st.markdown('* ##### Thriller')
                st.markdown('* ##### Western')
                st.markdown("From there, you could dive a bit deeper. Sub-genres were developed to give names and "
                            "expectations to certain types of films within each genre. The “thriller” genre, for "
                            "example, had the following sub-genres:")
                st.markdown('* ##### Crime Thriller')
                st.markdown('* ##### Disaster Thriller')
                st.markdown('* ##### Psychological Thriller')
                st.markdown('* ##### Techno Thriller')
                st.markdown("We have all of these genres on offer, so be adventurous and try something new.")


            if st.checkbox("What are the most popular genres?"):
                st.markdown("### How would you like to view this?")
                if st.checkbox("Bar Chart"):
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    fig = make_bar_chart(genre_df, 'Genre', title='Most Popular Movie Genres', xlab='Genre', ylab='Counts')
                    st.pyplot(fig)
                    st.markdown("Drama and comedy were the earliest genres of cinema, and they're still the most "
                            "popular genres today. Comedies entertain us by making us laugh, but dramas entertain "
                            "us by telling interesting stories. A good drama does more than just tell a good story,"
                            " however. It makes us care about the characters and feel many emotions. If something "
                            "good happens we feel like smiling, if something sad happens we feel like crying, and "
                            "if something bad is done to a character we like we might even feel angry. If we feel "
                            "emotions like these we'll get involved in the story and find it more entertaining. "
                            "But dramas can do even more than this. They can also make us think about important "
                            "issues and teach us important lessons about life and how to live."
                            "This is why drama and comedy remain the most watched movies on our platform.")

                if st.checkbox("Pie Chart"):
                    st.image('resources/genrespie.jpeg')
                    st.markdown("Drama and comedy were the earliest genres of cinema, and they're still the most "
                            "popular genres today. Comedies entertain us by making us laugh, but dramas entertain "
                            "us by telling interesting stories. A good drama does more than just tell a good story,"
                            " however. It makes us care about the characters and feel many emotions. If something "
                            "good happens we feel like smiling, if something sad happens we feel like crying, and "
                            "if something bad is done to a character we like we might even feel angry. If we feel "
                            "emotions like these we'll get involved in the story and find it more entertaining. "
                            "But dramas can do even more than this. They can also make us think about important "
                            "issues and teach us important lessons about life and how to live."
                            "This is why drama and comedy remain the most watched movies on our platform.")

        if st.sidebar.checkbox("Insights on Movie Titles"):
            st.markdown("### What would you like to see?")
            if st.checkbox("What are the most popular movies?"):
                st.image("resources/Most_Popular_movies_vs_ratings.png")
                st.markdown("### Lets have a look at the top 3: ")
                st.markdown("### 1. Shawshank Redemption")
                st.image("resources/ShawshankRedemption.jpg")
                st.markdown("The Shawshank Redemption is a 1994 American drama film written and directed by Frank "
                            "Darabont, based on the 1982 Stephen King novella Rita Hayworth and "
                            "Shawshank Redemption. It tells the story of banker Andy Dufresne (Tim Robbins), "
                            "who is sentenced to life in Shawshank State Penitentiary for the murders of his "
                            "wife and her lover, despite his claims of innocence. Over the following two decades, "
                            "he befriends a fellow prisoner, contraband smuggler Ellis 'Red' Redding "
                            "(Morgan Freeman), and becomes instrumental in a money-laundering operation "
                            "led by the prison warden Samuel Norton (Bob Gunton). William Sadler, Clancy "
                            "Brown, Gil Bellows, and James Whitmore appear in supporting roles.")
                st.markdown("### 2. Forest Gump")
                st.image("resources/Forrest_Gump_poster.jpg")
                st.markdown("Forrest Gump is a 1994 American drama film directed by Robert Zemeckis and "
                            "written by Eric Roth with comedic aspects. It is based on the 1986 novel of "
                            "the same name by Winston Groom and stars Tom Hanks, Robin Wright, Gary Sinise, "
                            "Mykelti Williamson and Sally Field. The story depicts several decades in the life "
                            "of Forrest Gump (Hanks), a slow-witted but kind-hearted man from Alabama who "
                            "witnesses and unwittingly influences several defining historical events in the "
                            "20th century United States. The film differs substantially from the novel.")

                st.markdown("### 3. Pulp Fiction")
                st.image("resources/Pulp_Fiction.jpg")
                st.markdown("Pulp Fiction is a 1994 American neo-noir black comedy crime film written and "
                            "directed by Quentin Tarantino, who conceived it with Roger Avary."
                            "[4] Starring John Travolta, Samuel L. Jackson, Bruce Willis, Tim Roth, "
                            "Ving Rhames, and Uma Thurman, it tells several stories of criminal Los Angeles. "
                            "The title refers to the pulp magazines and hardboiled crime novels popular during "
                            "the mid-20th century, known for their graphic violence and punchy dialogue.")


            if st.checkbox("How many movies were released over the years?"):
                st.image("resources/nummovies_released.png")
                st.markdown("Technological changes mean that movies are easier and cheaper to make and distribute."
                            "  In addition, shifts in the industry mean that films spend far less long in cinemas "
                            "before moving on to other platforms, such as DVD and Video On Demand. "
                            "Therefore it is much easier and more profitable to make movies in more "
                            "recent years as compared to earlier years. That explains the increase in "
                            "movie production.")

            if st.checkbox("What are the most popular buzz words used in movie titles?"):
                st.image("resources/Wordcloud_of_the_movie_titles.png")
                st.markdown("The title of a film can make or break the entire project, and not just during "
                            "the release. A poor, unfocussed, long, woolly and self important title can make "
                            "things hard right from the get go. The title of a screenplay or film serves one "
                            "purpose - To get the reader / actor / producer / investor / distributor / the final "
                            "consumers… all to say, 'tell me more'. Better still, 'I am in'. The title is a sales tool."
                            " It’s not a creative expression. They are in the business of selling their stories, "
                            "at every stage of the process. All art that has any commercial aspiration plays by "
                            "these rules.")

            if st.checkbox("What were the most expensive movies made?"):
                st.image("resources/most_expensive_movies.png")
                st.markdown("This graph is in tenths of billions. Meaning May Way has a budget of approximately "
                            "R30 Million Rands which sounds absurd but its $24 Million Dollars (which is still an "
                            "absurd amount of money to spend on a movie).May Way was promoted as the most expensive"
                            " Korean blockbuster yet produced. It has had a handful of cinema screenings in the UK"
                            " courtesy of the Terracotta Film Festival and a most loved blockbuster by Koreans. "
                            "1.3 Billion Rands. What is most interesting is that all the top 10 movies are Korean"
                            " movies which shows how much money Koreans are willing to spend on movie production "
                            "and how much revenue they are expecting to regain, especially from their Korean "
                            "supporters.")

        if st.sidebar.checkbox("Insights on Cast and Directors"):
            st.markdown("### What would you like to see")
            if st.checkbox("Who are the most popular actors?"):
                st.image("resources/Popular_Cast.png")
                st.markdown('### The top 3 most popular actors are Nicholas Cage, Johnny Depp '
                            'and Robert de Niro.')
                st.markdown('### Here are some reviews:')
                st.markdown("### Nicholas Cage")
                st.image("resources/Nicolas_Cage.jpg")
                st.markdown(" Nicolas Kim Coppola (born January 7, 1964),[2][3] known professionally as Nicolas "
                            "Cage, is an American actor and filmmaker. Cage has been nominated for numerous major "
                            "cinematic awards, and won an Academy Award, a Golden Globe, and Screen Actors Guild "
                            "Award for his performance in Leaving Las Vegas (1995). He earned his second Academy "
                            "Award nomination for his performance as Charlie and Donald Kaufman in Adaptation "
                            "(2002).During his early career, Cage starred in a variety of films such as Rumble "
                            "Fish (1983), Racing with the Moon (1984), Peggy Sue Got Married (1986), Raising "
                            "Arizona (1987), Vampire's Kiss (1989), Wild at Heart (1990), Honeymoon in Vegas "
                            "(1992), and Red Rock West (1993). During this period, John Willis' Screen World, "
                            "Vol. 36 listed him as one of twelve Promising New Actors of 1984.After winning his "
                            "Academy Award, Cage started starring in more mainstream films, such as The Rock "
                            "(1996), Con Air (1997), City of Angels (1998), 8mm (1999), Windtalkers (2002), "
                            "Lord of War (2005), The Wicker Man (2006), Bangkok Dangerous (2008) and Knowing "
                            "(2009). He also directed the film Sonny (2002), for which he was nominated for "
                            "Grand Special Prize at Deauville Film Festival. Cage owns the production company "
                            "Saturn Films and has produced films such as Shadow of the Vampire (2000) and The "
                            "Life of David Gale (2003). In October 1997, Cage was ranked No. 40 in "
                            "Empire magazine's The Top 100 Movie Stars of All Time list, while the next year, "
                            "he was placed No. 37 in Premiere's 100 most powerful people in Hollywood.In the "
                            "2010s, he starred in Kick-Ass (2010), Drive Angry (2011), Joe (2013), "
                            "The Runner (2015), Dog Eat Dog (2016), Mom and Dad (2017), Mandy (2018) and "
                            "Color Out of Space (2019). His participation in various film genres during this "
                            "time increased his popularity and gained him a cult following.")
                st.markdown("### Johnny Depp")
                st.image("resources/Johny_depp.jpg")
                st.markdown("John Christopher Depp II (born June 9, 1963) is an American actor, producer, and "
                            "musician. He has been nominated for ten Golden Globe Awards, winning one for "
                            "Best Actor for Sweeney Todd: The Demon Barber of Fleet Street (2007), and has "
                            "been nominated for three Academy Awards for Best Actor, among other accolades.Depp "
                            "made his debut in the horror film A Nightmare on Elm Street (1984), before rising to "
                            "prominence as a teen idol on the television series 21 Jump Street (1987–1990). In the "
                            "1990s, Depp acted mostly in independent films, often playing eccentric characters. "
                            "These included What's Eating Gilbert Grape (1993), Benny and Joon (1993), Dead Man (1995),"
                            "Donnie Brasco (1997) and Fear and Loathing in Las Vegas (1998). Depp also began collaborating "
                            "with director Tim Burton, starring in Edward Scissorhands (1990), Ed Wood (1994) and Sleepy "
                            "Hollow (1999).In the 2000s, Depp became one of the most commercially successful film stars by "
                            "playing Jack Sparrow in the swashbuckler film series Pirates of the Caribbean (2003–present). "
                            "He received critical praise for Finding Neverland (2004), and continued his commercially successful "
                            "collaboration with Tim Burton with the films Charlie and the Chocolate Factory (2005), Corpse Bride "
                            "(2005), Sweeney Todd (2007), and Alice in Wonderland (2010). In 2012, Depp was one of the world's "
                            "biggest film stars,[1][2] and was listed by the Guinness World Records as the world's highest-paid "
                            "actor, with earnings of US$75 million.[3] During the 2010s, Depp began producing films through his "
                            "company, Infinitum Nihil, and formed the rock supergroup Hollywood Vampires with Alice Cooper and Joe "
                            "Perry.")
                st.markdown("### Robert de Niro")
                st.image("resources/robert_deniro.jpg")
                st.markdown("Robert Anthony De Niro Jr was born August 17, 1943) is an American actor, producer, "
                            "and director. He is particularly known for his nine collaborations with filmmaker "
                            "Martin Scorsese, and is the recipient of various accolades, including two Academy "
                            "Awards, a Golden Globe Award, the Cecil B. DeMille Award, and a Screen Actors Guild "
                            "Life Achievement Award. In 2009, De Niro received the Kennedy Center Honor, and "
                            "received a Presidential Medal of Freedom from U.S. President Barack Obama in 2016.De "
                            "Niro portrayed Jake LaMotta in Scorsese's biographical drama Raging Bull (1980), and "
                            "won the Academy Award for Best Actor, his first in this category. He diversified to "
                            "other roles, playing a stand-up comic in The King of Comedy (1982), and gained "
                            "further recognition for his performances in Bernardo Bertolucci's epic 1900 (1976), "
                            "Sergio Leone's crime epic Once Upon a Time in America (1984), Terry Gilliam's "
                            "dystopian satire Brazil (1985), the religious epic The Mission (1986), and the "
                            "comedy Midnight Run (1988). De Niro portrayed gangster Jimmy Conway in Goodfellas "
                            "and a catatonic patient in the drama Awakenings (both 1990), and a criminal in the "
                            "psychological thriller Cape Fear (1991). All three films received praise for De "
                            "Niro's performances. He then starred in This Boy's Life (1993), and directed his "
                            "first feature film with 1993's A Bronx Tale. His other critical successes include "
                            "the crime films Heat and Casino (both 1995).He is also known for his comic roles in "
                            "Wag the Dog (1997), Analyze This (1999), and Meet the Parents (2000). After "
                            "appearing in several critically panned and commercially unsuccessful films, he "
                            "earned a Academy Award nomination for Best Supporting Actor for his performance "
                            "in David O. Russell's 2012 romantic comedy Silver Linings Playbook. In 2017, De "
                            "Niro portrayed Bernie Madoff in The Wizard of Lies, earning a Primetime Emmy Award "
                            "nomination. He then starred in the psychological thriller Joker and Scorsese's crime "
                            "epic The Irishman (both 2019).")

            if st.checkbox("Who are the most popular movie directors?"):
                st.image("resources/Top_10_Most_Popular_Movie_directors.png")
                st.markdown("### Luc Besson, Woody Allen and Stephen King are the directors who directed the most movies.")
                st.markdown("### Here are some reviews:")
                st.markdown("Number one on our list is Luc Besson. Luc Paul Maurice Besson is a French film "
                            "director, screenwriter, and producer. He directed and produced the films Subway"
                            " (1985), The Big Blue (1988), and La Femme Nikita (1990). Besson is associated "
                            "with the Cinéma du look film movement. He has been nominated for a César Award "
                            "for Best Director and Best Picture for his films Léon: The Professional and The "
                            "Messenger: The Story of Joan of Arc. He won Best Director and Best French Director"
                            " for his sci-fi action film The Fifth Element (1997). He wrote and directed the "
                            "2014 sci-fi action film Lucy and the 2017 space opera film Valerian and the City "
                            "of a Thousand Planets.")
                st.markdown("Woody Allen is an American film director, writer, actor, comedian, and musician, "
                            "whose career spans more than six decades and multiple Academy Award-winning films.")
                st.markdown("Stephen King has received Bram Stoker Awards, World Fantasy Awards, and British "
                            "Fantasy Society Awards. In 2003, the National Book Foundation awarded him the "
                            "Medal for Distinguished Contribution to American Letters. He has also received "
                            "awards for his contribution to literature for his entire bibliography, such as the "
                            "2004 World Fantasy Award for Life Achievement and the 2007 Grand Master Award from "
                            "the Mystery Writers of America.")

    if page_selection == "Box Office Trailers":
        st.title("Here are some trailers from the Box Office Top 10")
        st.markdown("Happy watching!!")
        st.markdown("### 1. Space Jam: A new Legacy")
        st.video("resources/spacejam.mov")
        st.markdown("Welcome to the Jam! NBA champion and global icon LeBron James goes on an epic adventure "
                    "alongside timeless Tune Bugs Bunny with the animated/live-action event 'Space Jam: A New"
                    " Legacy', from director Malcolm D. Lee and an innovative filmmaking team including Ryan "
                    "Coogler and Maverick Carter. This transformational journey is a manic mashup of two worlds"
                    " that reveals just how far some parents will go to connect with their kids. When LeBron "
                    "and his young son Dom are trapped in a digital space by a rogue A.I., LeBron must get them "
                    "home safe by leading Bugs, Lola Bunny and the whole gang of notoriously undisciplined Looney"
                    " Tunes to victory over the A.I.'s digitized champions on the court: a powered-up roster of "
                    "professional basketball stars as you've never seen them before. It's Tunes versus Goons in "
                    "the highest-stakes challenge of his life, that will redefine LeBron's bond with his son and "
                    "shine a light on the power of being yourself.")
        st.markdown("### 2. Black Widow")
        st.video("resources/blackwidow.mov")
        st.markdown("In Marvel Studios' action-packed spy thriller 'Black Widow', Natasha Romanoff aka Black"
                    " Widow confronts the darker parts of her ledger when a dangerous conspiracy with ties "
                    "to her past arises. Pursued by a force that will stop at nothing to bring her down, "
                    "Natasha must deal with her history as a spy and the broken relationships left in her "
                    "wake long before she became an Avenger. Scarlett Johansson reprises her role as Natasha/Black"
                    " Widow, Florence Pugh stars as Yelena, David Harbour portrays Alexei/The Red Guardian, "
                    "and Rachel Weisz is Melina. Directed by Cate Shortland and produced by Kevin Feige, "
                    "'Black Widow'- the first film in Phase Four of the Marvel Cinematic Universe- hits U.S. "
                    "theaters on May 1, 2020.")
        st.markdown("### 3.Escape Room: Tournament of Champions")
        st.video('resources/escaperoom.mov')
        st.markdown("Escape Room: Tournament of Champions is the sequel to the box office hit psychological "
                    "thriller that terrified audiences around the world. In this installment, six people "
                    "unwittingly find themselves locked in another series of escape rooms, slowly uncovering "
                    "what they have in common to survive…and discovering they’ve all played the game before.")
        st.markdown("### 4.F9")
        st.video("resources/f9.mov")
        st.markdown("Vin Diesel's Dom Toretto is leading a quiet life off the grid with Letty and his son, "
                    "little Brian, but they know that danger always lurks just over their peaceful horizon. "
                    "This time, that threat will force Dom to confront the sins of his past if hes going to "
                    "save those he loves most. His crew joins together to stop a world-shattering plot led by "
                    "the most skilled assassin and high-performance driver they've ever encountered: a man who "
                    "also happens to be Dom's forsaken brother, Jakob..")

        st.markdown("### 5. The Boss Baby: Family Business")
        st.video("resources/bossbaby.mov")
        st.markdown("In the sequel to DreamWorks Animation’s Oscar®-nominated blockbuster comedy, the Templeton "
                    "brothers—Tim (James Marsden, X-Men franchise) and his Boss Baby little bro Ted "
                    "(Alec Baldwin)—have become adults and drifted away from each other. Tim is now a "
                    "married stay-at-home dad. Ted is a hedge fund CEO. But a new boss baby with a cutting-edge "
                    "approach and a can-do attitude is about to bring them together again … and inspire a new "
                    "family business.")

        st.markdown("### 6. The Forever Purge")
        st.video("resources/purge.mov")
        st.markdown("All the rules are broken as a sect of lawless marauders decides that the annual Purge does "
                    "not stop at daybreak and instead should never end. Vaulting from the record-shattering "
                    "success of 2018's 'The First Purge', Blumhouse's infamous terror franchise hurtles into "
                    "innovative new territory as members of an underground movement, no longer satisfied with "
                    "one annual night of anarchy and murder, decide to overtake America through an unending "
                    "campaign of mayhem and massacre. No one is safe.")

        st.markdown("### 7. A Quiet Place Part II")
        st.video("resources/quiteplace.mov")
        st.markdown("Following the deadly events at home, the Abbott family (Emily Blunt, Millicent Simmonds, "
                    "Noah Jupe) must now face the terrors of the outside world as they continue their fight for "
                    "survival in silence. Forced to venture into the unknown, they quickly realize that the "
                    "creatures that hunt by sound are not the only threats that lurk beyond the sand path.")

        st.markdown("### 8. Roadrunner: A Film About Anthony Bourdain")
        st.video('resources/roadrunner.mov')
        st.markdown("It’s not where you go. It’s what you leave behind . . . "
                 "Chef, writer, adventurer, provocateur: Anthony Bourdain lived his life unabashedly. "
                 "Roadrunner: A Film About Anthony Bourdain is an intimate, behind-the-scenes look at how an "
                 "anonymous chef became a world-renowned cultural icon. From Academy Award®-winning filmmaker "
                 "Morgan Neville (20 Feet From Stardom, Won’t You Be My Neighbor?), this unflinching look at "
                 "Bourdain reverberates with his presence, in his own voice and in the way he indelibly "
                 "impacted the world around him.")

        st.markdown("### 9. Cruella")
        st.video("resources/cruella.mov")
        st.markdown("Academy Award winner Emma Stone ('La La Land') stars in Disney's 'Cruella,' an all-new "
                    "live-action feature film about the rebellious early days of one of cinemas most notorious "
                    "- and notoriously fashionable - villains, the legendary Cruella de Vil. 'Cruella', which is "
                    "set in 1970s London amidst the punk rock revolution, follows a young grifter named Estella, "
                    "a clever and creative girl determined to make a name for herself with her designs. She "
                    "befriends a pair of young thieves who appreciate her appetite for mischief, and together "
                    "they are able to build a life for themselves on the London streets. One day, Estella's "
                    "flair for fashion catches the eye of the Baroness von Hellman, a fashion legend who is "
                    "devastatingly chic and terrifyingly haute, played by two-time Oscar winner Emma Thompson "
                    "('Howards End', 'Sense & Sensibility'). But their relationship sets in motion a course of "
                    "events and revelations that will cause Estella to embrace her wicked side and become the "
                    "raucous, fashionable and revenge-bent Cruella.")

        st.markdown("### 10. PIG")
        st.video("resources/pig.mov")
        st.markdown("A truffle hunter who lives alone in the Oregonian wilderness must return to his past in "
                    "Portland in search of his beloved foraging pig after she is kidnapped.")






    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image("resources/streaming-services.jpg")
        st.markdown("Recommendation algorithms are at the core of any movie streaming service. They provide users "
                    "with personalized suggestions to reduce the amount of time and frustration to find "
                    "great content to watch. Streaming services such as Netflix, Showmax and Vimeo all use "
                    "recommender services")
        st.markdown("we have built recommendation engines which were based on different techniques and ideas."
                    "We have found a certain amount of success using the SVD Machine learning algorith"
                    " which utilised Hyperparameter "
                    "tuning to control the learning process. The success of our model was influenced by various"
                    " factors including the way we preprocessed our data and how we tuned the parameters to "
                    "ensure that we choose a set of optimal hyperparameters for our learning algorithm.")
        st.markdown("Furthermore, we put an enormous of effort and detail into our Exploratory Data Analysis to "
                    "assist us in determining the best possible ways to manipulate our data allowing us to deliver"
                    " optimal interpretation into our dataset and determine optimal factor settings.")
        st.markdown("We found it highly valuable to answer questions which we anticipate most movie platforms to "
                    "have to ensure ultimate satisfaction to their users. Questions such as “ Who are the most "
                    "popular actors?’,’Which genres get the best ratings?’ would allow platforms to gain "
                    "valuable insight and would allow for improvement.")
        st.markdown("With these means we built a model-driven app which provided greater context to the problem "
                    "and attempted solutions through additional application pages. This app would allow one to "
                    "enter three of their favourite movies and accurately predict how a user will rate a movie "
                    "they have not yet viewed based on their historical preferences.")
        st.markdown("With all these means and by building on crucial knowledge through research our team has "
                    "contributed to finding a solution to help movie viewers to be exposed to content they "
                    "would like to view or purchase , increase engagement with movie viewers and platforms; "
                    "and deliver more personalized customer experiences.")

    if page_selection == "About Us":
        st.title("About Us")
        st.markdown("## We are Team ZM4!")
        st.markdown("We are a team of budding Data scientists who are passionate about creating machine learning "
                    "products that can be used by everyday people. Our goal is to create products that actually "
                    "make a difference in the everyday  lives of normal people.")
        st.markdown("Team ZM4 was challenged by EDSA(Explore Data Science Academy), with the task of constructing "
                    "a recommendation algorithm based on content or collaborative filtering, capable of accurately "
                    "predicting how a user will rate a movie they have not yet viewed based on their historical "
                    "preferences.")
        st.markdown("We feel that we have found the best solution for this challenge.")
        st.markdown("Please check out our GitHub repo for a more detailed Jupyter Notebook detailing how we "
                    "got to our winning solution.")
        st.markdown("https://www.kaggle.com/c/edsa-movie-recommendation-challenge/code?competitionId=27685&sortBy=dateRun&tab=profile")
        st.title("Our Team")
        st.markdown("### Thato Bogopane : App Development and Modelling")
        st.image("resources/Thato.png")
        st.markdown("### Keletso Pule : Data engineering and Data Analysis")
        st.image("resources/Keletso.png")
        st.markdown("### Morgan : Data Engineering and Modelling")
        st.image("resources/Morgan.png")
        st.markdown("### Herpelot: Data Engineering and Analysis")
        st.image("resources/Pontso.png")



    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()

# Reddit-Comment-Analysis-Project

##Project analyzing reddit comments of the Notre Dame football football subreddit and general /CFB rubreddit using sentiment analysis and topic modeling. 

This project utilizes the Arctic Shift API to scrape reddit comment data, including the date posted(in unix), upvotes, poster username, main post information and mos importantly the comment text. There is no authorization required but there is a 100 post API limit that can be worked around. My analyis focused on comparing the reactions of the general /CFB subreddit and the /NotreDameFootball specific subreddit to both Notre Dame being left out of the playoff and being upset by Northern Illinois. 

You can adjust the parameters of the two get_reddit_posts_comments method calls to compare any two subreddits reaction to the event of your choosing, although custom stopwords will have to be adjusted as well to do topic modeling. The start and end date parameters are taken as unix timestamps. 

```
def get_reddit_posts_comments(subreddit, search, total_posts=500, comments_per_post=50, start_date=None, end_date=None):
```


```
df_cfb = get_reddit_posts_comments(subreddit="CFB", search="Notre Dame", total_posts=100, 
start_date=1765065600,end_date=1765151999)
df_nd = get_reddit_posts_comments(subreddit="notredamefootball", search="Notre Dame", total_posts=400,
comments_per_post=100, start_date=1764997200,end_date=1765256399)
```
After obtaining the reddit comment data and cleaning the comments, the project begins to perform sentiment analysis with spaCy. It assigns each comment a polarity and subjectivity score which is added to the master dataframe. After running the sentiment analysis I compared the results with dendsity plots, one of which(NIU) is shown below. When comparing the two there was a lot more extreme emotions on both sides for the NIU game, most likely because of how shocking it was even in comparison with unexpectedly missing the playoff.

<img width="415" height="307" alt="Screenshot 2026-03-02 at 1 24 30 AM" src="https://github.com/user-attachments/assets/f575f55d-e8f6-49ad-84e3-b85d1a3da50b" />

I then performed another form of sentiment analsis using a transformer model to detect emotion. Each comment was assigned a primary and secondary emotion as well as a score that was added to the master dataframe. I compared the results using two boxplots that showed type of emotion vs how upvoted the comment was, with outliers removed. The NIU boxplot is shown below. There was extremely high rates of upvotes in joyful comments in the general subredit displaying how the general public is usually ecstatic when Notre Dame embarreses themselves There is also a high degree of fear comments upvoted in the Notre Dame subreddit that wasn't present in the playoff boxplot. Fearful comments were actually somewhat downvoted in the playoff visualization. I interpreted this as after the NIU loss Notre Dame fans were fearful about the direction the program was heading, had major questions about Marcus Freeman, and were frustrated with endless transfer QBs. We don't see that same fear after they miss the 2025 playoff because their 2024 playoff run gave confidence to fans that although the playoff snub hurt, it's only a matter of time until they win that elusive national title. 

<img width="491" height="353" alt="Screenshot 2026-03-02 at 1 29 38 AM" src="https://github.com/user-attachments/assets/889e27d9-a0d1-4fab-8848-20776a4d79c0" />


The final step of the project was to perform LDA topic modeling on the entire master dataframe of comments. First the data needed to broken down to a by word level rather then by comment. Next stopwords were removed, the data was lemmatized, and bigrams were created. After the LDA model was fit the results of the topic modeling were visualized dynamically with pyLDAvis. As mentioned above the custom stopwords must be adjusted to run the LDA topic modeling on a topic of you own choosing. Below is screenshot of the NIU pyLDAvis visualization. 

<img width="620" height="473" alt="Screenshot 2026-03-02 at 1 37 37 AM" src="https://github.com/user-attachments/assets/9de8380f-6de3-4bfd-8920-eca4d1131cd4" />



# Reddit-Comment-Analysis-Project
Project analyzing Notre Dame football reddit comments
##Project analyzes reddit comments of the Notre Dame football football subreddit and general /CFB rubreddit using sentiment analysis and topic modeling. 

This project utilizes the Arctic Shift API to scrape reddit comment data, including the date posted(in unix), upvotes, poster username, main post information and mos importantly the comment text. There is no authorization required but there is a 100 post API limit that can be worked around. My analyis focused on comparing the reactions of the general /CFB subreddit and the /NotreDameFootball specific subreddit to both Notre Dame being left out of the playoff and being upset by Northern Illinois. 

You can adjust the parameters of the two get_reddit_posts_comments method calls to compare any two subreddits reaction to the event of your choosing, although custom stopwords will have to be adjusted as well to do topic modeling. The start and end date parameters are taken as unix timestamps. 

'''
def get_reddit_posts_comments(subreddit, search, total_posts=500, comments_per_post=50, start_date=None, end_date=None):
'''


'''
df_cfb = get_reddit_posts_comments(subreddit="CFB", search="Notre Dame", total_posts=100, 
start_date=1765065600,end_date=1765151999)
df_nd = get_reddit_posts_comments(subreddit="notredamefootball", search="Notre Dame", total_posts=400,
comments_per_post=100, start_date=1764997200,end_date=1765256399)
'''



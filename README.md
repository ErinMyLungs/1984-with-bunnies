For your three preliminary proposals, you need a brief description (one paragraph), and to answer four questions.

1) Who will use the results of the project? Be specific. It’s ok to have multiple different types of people with different problems, but you need at least one.

Examples: 
People considering a round-the-world trip
Farmers who wish to reduce pesticide use
Anyone who enjoys viewing fall foliage
People who have trouble matching clothes

2) How will your project solve the users’ problem, or help them in some way? It’s ok if your capstone is just a proof of concept as long as future version could be useful. It doesn’t have to help them a lot or to make money, just solve some sort of problem they have. In particular, it’s not enough that the project is interesting or cool.

3) Where will you get data that will let you solve the problem? You don’t need to verify that all the data you need is available for the preliminary proposal, just have a reasonable expectation that you can get it.

4) How will you evaluate your project? If you’re building a predictive model this is trivial: cross validation. If you’re recommending round-the-world trips it might be more difficult.


Subreddit recommender - the algo.

1. Finding subreddits is garbage. Matching up with users/subreddits based on descriptions is also problematic.
    * Look at user activity in subreddits
    * look at sentiment of average content of subreddit?
        * a month of scraping text posts/content?
        * ratio of pics to text
        * average activity?
    * Use prior subreddit subscriptions and look at similarities of other redditors?
        * Similarity/recommenders?
    * Improving recommendations can improve user activity, possibly lower churn, drive user content creation
        * More cash money

1. PSAW should be realtively easy
    * subreddit subscription information is not publicly available
        * where users comment? (Not a good indicator)
        
1. Artificially drop a subredditand test if the algo recommends it back? 
#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc

#create Spark context
sc = SparkContext()

#create Spark Streaming context and SQL context to execute sql queries
ssc = StreamingContext(sc, 10 )
sqlContext = SQLContext(sc)

#create a socket stream to connect to the socket created by TweetRead.py
socket_stream = ssc.socketTextStream("127.0.0.1", 9999)

# take in the lines in a window of 20 
lines = socket_stream.window( 20 )


# create tuples, named tuple has fields
from collections import namedtuple

# we want to count the actual hash tags in the tweets
fields = ("tag", "count" )
Tweet = namedtuple( 'Tweet', fields )


# we take the lines and we save the top 10 tags and their counts in a sql table
# we later use pandas and matplotlib to show the content of this small table - refreshed every 3 sec
( lines.flatMap( lambda text: text.split( " " ) ) #Splits to a list
  .filter( lambda word: word.lower().startswith("#") ) # Checks for hashtag calls
  .map( lambda word: ( word.lower(), 1 ) ) # Lower cases the word
  .reduceByKey( lambda a, b: a + b ) # Reduces by key we get the word count of the hash tags
  .map( lambda rec: Tweet( rec[0], rec[1] ) ) # Stores in a Tweet Object
  .foreachRDD( lambda rdd: rdd.toDF().sort( desc("count") ) # Sorts Them in a DF
  .limit(10).registerTempTable("tweets") ) ) # Registers to a table.

#before running this line we have to run the python srcipt TweetRead.py in the other terminal
ssc.start() 

# draw the content of the small table
import time
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns

#draw top 10 hash tags and their count. The barplot resfreshes every 3 seconds
count = 0
while count < 10:
    
    time.sleep( 3 )
    top_10_tweets = sqlContext.sql( 'Select tag, count from tweets' )
    top_10_df = top_10_tweets.toPandas()
    display.clear_output(wait=True)
    sns.plt.figure( figsize = ( 10, 8 ) )
    sns.barplot( x="count", y="tag", data=top_10_df)
    sns.plt.show()
    count = count + 1

# to run when we want to stop the process
ssc.stop()






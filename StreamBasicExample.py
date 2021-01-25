#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

#spark context with two local working threads
sc = SparkContext("local[2]", "NetworkWordCount")

#streaming context with batch interval = 1 sec
ssc = StreamingContext(sc,1)

#from the streaming context create a DStream 
#that will connect to a host - localhost and port 9999
lines = ssc.socketTextStream('localhost', 9999)

# split the lines which is a srting into words
words = lines.flatMap(lambda line:line.split(' '))

# map -create the tupples
pairs = words.map(lambda word:(word,1))

# reduce - group by key - the key is the word
word_counts = pairs.reduceByKey(lambda num1, num2:num1+num2)

# print as the words count themselves
word_counts.pprint()

# start the process
ssc.start()

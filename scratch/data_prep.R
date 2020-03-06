
rm(list = ls())
options(stringsAsFactors = FALSE)

setwd('~/Desktop/Projects/Active/SIOP/2020/siop-2020-text-mining-and-nlp')

library(magrittr)

# data_original <- read.csv('data/siop_2020_txt_lo-wide.csv')

# text_pro <- data_original$txt_pro %>%
#             gsub(pattern     = 'Pros</b><br/>', 
#                  replacement = '', 
#                  x           = .) %>%
#             data.frame(Text = ., 
#                        Pro  = 1)

# text_con <- data_original$txt_con %>%
#             gsub(pattern     = 'Cons</b><br/>', 
#                  replacement = '', 
#                  x           = .) %>%
#             gsub(pattern     = '<br/><br/>', 
#                  replacement = '', 
#                  x           = .) %>%
#             data.frame(Text = ., 
#                        Pro  = 0)

# # combine
# text_dat <- rbind(text_pro, text_con)

# write.csv(x         = text_dat, 
#           file      = 'data/text_plus_pro_con.csv', 
#           row.names = FALSE)


# not sure if we want the above as part of the tutorial, or if we should just
# have the data ready to go like that... leaving it up there for us to decide
# if we have enough time i suppose


library(tm)
library(magrittr)

# create a corpus per row of text in the data. a corpus is a format for storing
# text data.. often contains meta information (but doesn't have too)

# VectorSource - tells R to treat each element as if it were a document
# SimpleCorpus - function that turns the text in to corpora

text_dat    <- read.csv('data/text_plus_pro_con.csv')

text_corpus <- VectorSource(text_dat$Text) %>%
               VCorpus()


# inspecting the first element

# $content - shows the text
# $meta    - shows the meta data (if any exists)

text_corpus[['1']]$content
text_corpus[['1']]$meta


# now that we have our data in format that is useable by tm (and other packages),
# we will start to clean the data and do a little pre-processing

# transform every character to lower case
text_corpus %<>% tm_map(content_transformer(tolower))

# remove numbers from the text
text_corpus %<>% tm_map(removeNumbers)

# remove punctuation
text_corpus %<>% tm_map(removePunctuation)

# remove stop words. stop words are basically words that are of little value..
# e.g., 'a', 'the', 'I', 'me', etc. You can see the stopwords that are removed
# by typing stopwords('english')
text_corpus %<>% tm_map(removeWords, stopwords('english'))

# stripping any additional white space - multiple whitespace characters are 
# collapsed to a single blank.
text_corpus %<>% tm_map(stripWhitespace)

# create a dictionary of the text_corpus before stemming so that we can
# reconstruct the first version of the words - if we don't take the words out
# of the corpus, completing the stem words will not take into consideration
# how often the words appear inthe corpus.
text_dict     <- unlist(lapply(text_corpus, words))

# stemming. Stemming is the process of transforming words to a common root. For
# example, 'likes', 'liked', 'likely', and 'liking' will all be transformed to 
# 'like'
text_corpus %<>% tm_map(stemDocument)


# getting the data ready for graphing / basic predictive modeling

# creating a Document Term matrix
# A document term matrix is a matrix whose rows are represent an individual 
# document (row of text in our case) and whose columns represent individual
# words. A cell then shows the frequency a word shows up in a particular 
# document

text_dtm <- DocumentTermMatrix(text_corpus)

# as a heads up, R does not store this matrix as a typical matrix. If you type
# in the below you won't get what you expect
text_dtm[1:3, 1:10]

# however, if you type the below you will get something that looks familiar
inspect(text_dtm[1:3, 1:10])

# we can take the document term matrix, and compute the overall word frequency.
# This information can by handy for a wordcloud representation of the data
text_total_freq <- colSums(as.matrix(text_dtm)) %>%
                   as.data.frame() %>%
                   set_colnames('count') %>%
                   .[order(.$count, decreasing = TRUE), , drop = FALSE]

head(text_total_freq)

#
# Sentiment Analysis
library(SentimentAnalysis)

# Using the Harvard-IV dictionary (General Inquirer) 
# which is a dictionary of words associated with positive (1,915 words) or 
# negative (2,291 words) sentiment.
# this next line takes 1-2 mins to run... we will run on a subset
# text_sent <- analyzeSentiment(text_dtm, language = "english")
text_sent <- analyzeSentiment(text_dtm[1:1000, ], language = "english")

# were going to just select the Harvard-IV dictionary results ..  
# there are other dictionaries used for sentiment as well
# type ?DictionaryHE and ?DictionaryLM for details on other dictionaries
text_sent %<>% .[, 1:4] %>%
               as.data.frame()

# here we see four variables:
#   * WordCount - number of words in the corpus / string of text
#   * SentimentGI - overall sentiment (positive - negative)
#   * NegativityGI - negative sentiment; higher = more negative
#   * PositivityGI - positive sentiment; higher = more positive
head(text_sent)

summary(text_sent$SentimentGI)

# look at the most positive corpora
positive_order <- order(text_sent$SentimentGI, decreasing = TRUE)
text_dat[positive_order[1:5], ]

# Emotional word categorization using NRC emotion lexicon. For details
# see: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
# the NRC emotion lexicon associates words with 8 emotions: anger, fear, 
# anticipation, trust, surprise, sadness, joy, and disgust (as well as
# negative and positive sentiment)
library(syuzhet)

# running this next line of code takes like 4ish minutes... so we will run
# the nrc function on a subset of the data
# text_emot <- get_nrc_sentiment(char_v = text_dat$Text)
text_emot <- get_nrc_sentiment(char_v = text_dat$Text[1:1000])

# inpsecting the data, we see a column per emtion and sentiment; the number
# is a count. For example, there are 2 words in sentence 3 that are associated
# with the emotion Trust
head(text_emot)

# seems that trust is the dominant emotion in these comments
colSums(text_emot)


# Basic predictive modeling setup
# bag of words approach and n-gram approach

# Bag of words approach basically ignores sentence grammer, word order, etc. 
# How we have set up our data thus far is inline w/ the bag of words approach.
# We have created Term Frequency matrix - often times for modeling, we create 
# something called the Term Frequency - Inverse Document Frequency Matrix. This
# matrix weights Term Frequency by how prevalent they are in the corpora (
# text strings). If the word shows up often in and across the documents - they 
# get less weight (for example stop words - however we have already removed
# these). A good overview can be found at: http://www.tfidf.com/
text_tfidf <- DocumentTermMatrix(x       = text_corpus, 
                                 control = list(weighting = weightTfIdf)) %>%
              as.matrix()


# you could run this whole model - but it takes some time.. we will run a 
# smaller one as an example
# mod_data_1 <- data.frame(text_tfidf, y = text_dat$Pro)
# summary(glm(y ~ ., data = mod_data_1))
mod_data_1 <- data.frame(text_tfidf[, 1:50], y = text_dat$Pro)
mod_1      <- glm(y ~ ., data = mod_data_1)

summary(mod_1)

# n-gram
# the previous approach looked at words independtly of other words w/in a text
# string. N-grams allow you to create pair; triplets; etc representations of 
# words. So for example, if we had the sentence:
#   - "Great developer of people, managers allowed discretion"
# a 2-gram representation (tokenization) would be:
#   - Great developer, developer of, of people, people managers, 
#     managers allowed, allowed discretion
# a 3-gram representation would b:
#   - Great developer of, developer of people, of people managers,
#     people managers allowed, managers allowed discretion
# etc.

# we will create a custom Tokenizer function to pass to tm's functions
ngram_tokenizer <- function(x, n) {
  # turn a text string (corpus) into a vector of words
  text_words <- words(x)
  # create a n-gram representation of the vector of words
  text_gram  <- ngrams(text_words, n = n)
  # format the text_gram in a way that the tm package functions can process
  out        <- lapply(text_gram, paste, collapse = ' ') %>%
                unlist()

  return(out)
}

bigram  <- function(x) ngram_tokenizer(x, n = 2)
trigram <- function(x) ngram_tokenizer(x, n = 3)

# shrink the size of the text corpus object - for demo only
# running this on all the data may cause an error due to memory limits. 
text_corpus_shrunk <- text_corpus[1:1000]

bigram_tfidf <- DocumentTermMatrix(x       = text_corpus_shrunk, 
                                   control = list(tokenize  = bigram,
                                                  weighting = weightTfIdf)) %>%
                as.matrix()

bigram_tfidf[1:10, 1:5]

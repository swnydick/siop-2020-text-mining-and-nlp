
rm(list = ls())
options(stringsAsFactors = FALSE)

setwd('~/Desktop/Projects/Active/SIOP/2020/siop-2020-text-mining-and-nlp')

library(magrittr)

data_original <- read.csv('data/siop_2020_txt_lo-wide.csv')

text_pro <- data_original$txt_pro %>%
            gsub(pattern     = 'Pros</b><br/>', 
                 replacement = '', 
                 x           = .) %>%
            data.frame(Text = ., 
                       Pro  = 1)

text_con <- data_original$txt_con %>%
            gsub(pattern     = 'Cons</b><br/>', 
                 replacement = '', 
                 x           = .) %>%
            gsub(pattern     = '<br/><br/>', 
                 replacement = '', 
                 x           = .) %>%
            data.frame(Text = ., 
                       Pro  = 0)

# combine
text_dat <- rbind(text_pro, text_con)

write.csv(x         = text_dat, 
          file      = 'data/text_plus_pro_con.csv', 
          row.names = FALSE)


# not sure if we want the above as part of the tutorial, or if we should just
# have the data ready to go like that... leaving it up there for us to decide
# if we have enough time i suppose


library(tm)

# create a corpus per row of text in the data. a corpus is a format for storing
# text data.. often contains meta information (but doesn't have too)

# VectorSource - tells R to treat each element as if it were a document
# SimpleCorpus - function that turns the text in to corpora

text_corpus <- VectorSource(text_dat$Text) %>%
               SimpleCorpus()


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








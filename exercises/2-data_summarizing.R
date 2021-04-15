#########################################
# NLP Data Summarizing                  #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2021-04-15                            #
#########################################

# 1. Setup / Required Packages =================================================

# Make sure to have objects from data_prep.R loaded for this section. We will
# use the following packages
library(tm)
library(wordcloud)
library(wordcloud2)
library(tidytext)
library(dplyr)
library(tidyr)
library(ggplot2)

# get project directory (SPECIFY MANUALLY???)
proj_dir     <- here::here()

setwd(proj_dir)
analyses_dir <- file.path(proj_dir, "exercises")

# 2. Wordclouds ================================================================

# Using the above we can start building displays of the data.

# If we have a frequence data.frame (with words on the rows and a single column
# of counts, we can plot a simple word.cloud.
wordcloud_colors       <- c("#339966", "#3399CC", "#CC6600", "#660066", "#990000",
                            "#FFCC00", "#CE4040", "#98C7FF", "#FFC80A", "#7AB0C2",
                            "#3A456A")

# The oldest wordcloud package in R can display a simple circle wordcloud
# (as well as comparison word clouds) - let's take a subset of words (or it
# will take a while)
text_total_freq_part   <- text_total_freq[1:400, , drop = FALSE]

#   2a. Basic Wordclouds -------------------------------------------------------

# completing the stems so that the words are actual words that show up in the
# document rather than stemmed/partial words (slow, so pre-processing).
text_total_freq_part_prep <-
  text_total_freq_part %>%
  mutate(
    word  = stemCompletion(x          = rownames(.),
                           dictionary = text_dict)
  )

# reassign pre-processed part
text_total_freq_part      <- text_total_freq_part_prep

wordcloud(words        = text_total_freq_part$word,
          freq         = text_total_freq_part$count,
          min.freq     = 10,
          max.words    = 200,
          random.order = TRUE,
          random.color = FALSE,
          rot.per      = .3,
          colors       = wordcloud_colors)

# the wordcloud2 package allows you to use pictures to define wordclouds (using
# the wordcloud2.js library)
# - to make this function work
#   - you MUST have two columns: word and freq as the first two columns of your df
#   - word (and freq) MUST be unnamed vectors
text_total_freq_part2 <- rename(text_total_freq_part,
                                freq = count) %>%
                         select(word, freq) %>%
                         mutate(word = unname(word))
wordcloud2(data        = text_total_freq_part2,
           color       = wordcloud_colors,
           minRotation = -pi/2,
           maxRotation = +pi/2,
           rotateRatio = .3,
           shape       = "circle")

# you can change the shape of the wordcloud to a default shape
wordcloud2(data        = text_total_freq_part2,
           color       = wordcloud_colors,
           minRotation = -pi/4,
           maxRotation = +pi/4,
           rotateRatio = .3,
           shape       = "star")

# you can use a fancy shape that you create on your own
wordcloud2(data        = text_total_freq_part2,
           color       = wordcloud_colors[1],
           minRotation = -pi/4,
           maxRotation = +pi/4,
           rotateRatio = .3,
           figPath     = file.path(basename(analyses_dir), "man-standing.png"))

# OR letters
letterCloud(data        = text_total_freq_part2,
            word        = "S",
            color       = wordcloud_colors[1],
            wordSize    = 0,
            rotateRatio = .3)

# To make picture-based wordclouds work in Rstudio:
# - NEED to install wordcloud2 from github: devtools::install_github("lchiffon/wordcloud2")
# - NEED to click "refresh" after running the image
# - You can choose multiple letters for letter cloud, but sometimes it will not
#   include really big letters (for some odd reason)

#   2b. Sentiment-based Wordclouds ---------------------------------------------

# tidytext has a function to pull sentiments out for individual words, which we
# can use the summarize the data

# first completing words for ALL rows in total_text_freq matrix
# (next lines take several minutes to run, so will read in file already processed)
# text_total_freq_all   <- text_total_freq
# text_total_freq_all %<>% mutate(
#   word  = stemCompletion(x          = rownames(.),
#                          dictionary = text_dict)
# )
text_total_freq_all <- readRDS(file.path(analyses_dir, "text_total_frequencies.Rdata"))
sentiments          <- get_sentiments(lexicon = "bing")

# - merge word_frequencies with sentiment lists
# - add up count for each word (we could have multiple stems)
# - keep only negative and positive words
text_total_freq_all <- merge(x     = text_total_freq_all,
                             y     = sentiments,
                             by    = "word") %>%
                       group_by(word, sentiment) %>%
                       summarize(across(.cols = "count",
                                        .fns  = sum)) %>%
                       filter(sentiment %in% c("negative", "positive"))

# we could plot top ten negative and positive words
# - for each sentiment type (e.g., negative or positive)
# - pull 10 rows that have the top in the "count" column
# - reorder the word vector to have the HIGHEST factor equal to one with the
#   highest count (so that the plot is ordered correctly)
top_ten_sent       <- text_total_freq_all %>%
                      group_by(sentiment) %>%
                      top_n(n  = 10,
                            wt = count) %>%
                      ungroup() %>%
                      mutate(word = reorder(word, count))

# - making a plot divided by (and colored) by sentiment
#   (note that this code ONLY works on ggplot version 3.3.0 ... install new)
top_ten_sent_plot <- ggplot(data    = top_ten_sent,
                            mapping = aes(y    = word,
                                          x    = count,
                                          fill = sentiment)) +
                     geom_col(show.legend = FALSE) +
                     facet_wrap(facets = ~sentiment,
                                scales = "free_y") +
                     labs(x = "Most specified positive and negative words",
                          y = NULL) + 
                     scale_fill_manual(values = wordcloud_colors) +
                     theme_bw()
top_ten_sent_plot

# you can try the above code with a different sentiment dictionary (like afinn)

# we can actually create a sentiment based wordcloud as well
dtm_sentiment      <- text_total_freq_all %>%
                      pivot_wider(names_from  = "sentiment",
                                  values_from = "count",
                                  values_fill = list(count = 0)) %>%
                      tibble::column_to_rownames("word") %>%
                      as.matrix()

comparison.cloud(term.matrix  = dtm_sentiment,
                 max.words    = 300,
                 rot.per      = .3,
                 colors       = wordcloud_colors,
                 title.size   = 3,
                 match.colors = TRUE)

# We can try the exact same thing but with TWO changes:
# - Change lexicon from "bing" to "nrc" in get_sentiments
# - Remove the "filter" (plus the previous %>%) from text_total_freq_all
# This will plot ALL sentiments and not JUST the positve/negative ones!

#   2c. Type-based Wordclouds --------------------------------------------------

# One thing that might be interesting is to see the most common words of the
# "pros" and "cons" data:
# - split the text_corpus into "pros" and "cons"
# - create a DTM on each text corpus
# - use the DTM to create a total_freq df
text_corpus_type     <- split(x = text_corpus,
                              f = text_dat$Pro)
text_dtm_type        <- lapply(X   = text_corpus_type,
                               FUN = DocumentTermMatrix)
text_total_freq_type <- lapply(
  X   = text_dtm_type,
  FUN = function(dtm){
    data.frame(count = colSums(as.matrix(dtm))) %>%
    .[order(.$count, decreasing = TRUE), , drop = FALSE] %>%
    tibble::rownames_to_column(var = "word")
})

# - remove words that are the same in both pro and con
common_words           <- lapply(X   = text_total_freq_type,
                                 FUN = function(x) x$word) %>%
                          Reduce(f = intersect)

# - bind everything together and remove common types
text_total_freq_type   <- bind_rows(text_total_freq_type,
                                    .id = "type") %>%
                          mutate(type = c("con", "pro")[as.numeric(type) + 1]) %>%
                          filter(!(word %in% c("", common_words)))

# - taking the top 200 for each word type
text_total_freq_type %<>% group_by(type) %>%
                          top_n(n  = 200,
                                wt = count) %>%
                          ungroup()

# - completing the stems of each word (slow, so pre-processing)
text_total_freq_type_prep <- mutate(
  .data = text_total_freq_type,
  word  = stemCompletion(x          = word,
                         dictionary = text_dict)
)

# reassign pre-processed part
text_total_freq_type      <- text_total_freq_type_prep

# - combining different stems together
text_total_freq_type %<>% group_by(type, word) %>%
                          summarize(across(.cols = "count",
                                           .fns  = sum)) %>%
                          ungroup()

# - reshaping and creating rownames
dtm_type              <- text_total_freq_type %>%
                         pivot_wider(names_from  = "type",
                                     values_from = "count",
                                     values_fill = list(count = 0)) %>%
                         tibble::column_to_rownames("word") %>%
                         as.matrix()

# - we can make the same wordcloud we made earlier but split on type
comparison.cloud(term.matrix  = dtm_type,
                 max.words    = 300,
                 rot.per      = .3,
                 colors       = wordcloud_colors,
                 title.size   = 3,
                 match.colors = TRUE)

#   2d. Overall Sentiment Plots ------------------------------------------------

# we can make a plot of the text sentiment across the diffferent text

# let's take the first 1000 pros and cons of text_dat
inds          <- lapply(X   = c(0, nrow(text_dat) / 2),
                        FUN = function(i, n) i + seq_len(n),
                        n   = 1000)
inds          <- pmin(pmax(unique(unlist(inds)), 1), nrow(text_dat))

text_dat_sub  <- text_dat[inds, , drop = FALSE]
text_dtm_sub  <- text_dtm[inds, ]

# then we can analyze sentiment for those people
text_sent_sub <- analyzeSentiment(text_dtm_sub, language = "english")

# one thing to do is to check whether the "pros" and "cons" relate to the
# actual sentiment of the response
compareToResponse(sentiment = text_sent_sub,
                  response  = text_dat_sub$Pro)[ , 1:4] %>%
  round(1)

# - sensitivity/Specificity doesn't seem to be working
# - coding to binary response doesn't seem to be working either

# we can indicate indices for positive and negative Pro
text_dat_sub   %<>% group_by(Pro) %>%
                    mutate(id = seq_len(n())) %>%
                    ungroup()
text_sent_plot   <- as.data.frame(text_sent_sub) %>%
                    .[1:4] %>%
                    cbind(text_dat_sub, .) 

# selecting/renaming columns to make it easier to plot/facet/etc.
text_sent_plot %<>% transmute(id         = id,
                              type       = c("con", "pro")[Pro + 1],
                              text       = Text,
                              word_count = WordCount,
                              sentiment  = SentimentGI)



quant_sent_plot  <- ggplot(data    = text_sent_plot,
                           mapping = aes(x    = id,
                                         y    = sentiment,
                                         fill = type)) +
                    geom_col(show.legend = FALSE) +
                    facet_wrap(facets = ~type,
                               ncol   = 1,
                               scales = "free_y") +
                    labs(x = "person index") +
                    scale_fill_manual(values = wordcloud_colors) +
                    theme_bw()
  
quant_sent_plot

# all in all, people are fairly positive, and even their "cons" are positive
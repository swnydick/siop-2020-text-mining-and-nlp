
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
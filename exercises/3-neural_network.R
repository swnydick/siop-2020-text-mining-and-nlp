#########################################
# NLP Tensorflow                        #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2021-04-15                            #
#########################################

# 1. Setup / Required Packages =================================================

# Make sure to have objects from data_prep.R loaded for this section. We will
# use the following packages

# YES GPU #
# NOTE!!!!
# ONLY SET-gou IF YOU HAVE CONFIGURED CUDA AND CuDNN
# keras also has a function to install tf (try this first)
#keras::install_keras(tensorflow="2.2.0-gpu")

# if having issues, can help to bypass and go through tf directly
#tensorflow::install_tensorflow(version = "2.2.0-gpu")

# NO GPU #
#keras::install_keras()

# Check python/reticulate.
# May need to set interpreter in tools > global options > python
reticulate::py_config()
tensorflow::tf_config()
# Should see tensorflow on BOTH

require(data.table)
require(keras)
require(magrittr)

# 2. Prepare Data & Set params =================================================

# data from 1-data_prep.R, but read in standalone here

# Use fread for speed
data_dir   <- file.path("data")
dt         <- fread(file.path(data_dir, "text_plus_pro_con.csv"))

max_vocab  <- 8000 # vocabulary size (larger -> more thourough but slower)
max_length <- 80   # text cutoff at n (make bigger for longer texts at expense of size/time)

#   2a Clean text ==============================================================

# will need to convert to lower, also worth correcting a few common issues with acronyms, etc
# can use textclean too
# dt$text %<>%                               # function:
#   textclean::replace_non_ascii() %>%       # fix non standard character
#   #textclean::replace_emoji() %>%          # emojis aren't used here, but they may be!
#   textclean::replace_emoticon() %>%        # :) -> "happy"
#   textclean::replace_word_elongation() %>% # nooooooo -> no
#   textclean::replace_internet_slang() %>%  # lol -> laughing
#   textclean::replace_ordinal() %>%         # 1st -> "first
#   textclean::replace_kern() %>%            # kern spacing fix e.g. K E R N -> kern
#   textclean::replace_html() %>%            # <br> -> ""
#   tolower()

dt[, Text := gsub(",", " ,", Text)]
dt[, Text := gsub("^.*</b><br/>", "" , Text)]  # Remove pros/cons html tag
dt[, Text := gsub("<.*?>", "", Text)] # remove HTML junt
dt[, Text := gsub("brbr$", "" , Text)]  
dt[, Text := gsub("[\\\r\\\n]", " " , Text)] 
dt[, Text := tolower(Text)]
# target must be numeric to feed into neural net!
dt[, target := as.integer(Pro)]

# remove rows with NA values
dt <- dt[!is.na(dt$Text) & !is.na(dt$target), .(Text, target)]


#   2b Tokenize ================================================================

# To feed words into a neural network, they need to be made into numbers somehow so math can happen
# e.g.
# "I like dogs"
# could be a sparse matrix
# I | am | to | so | like | this | cats | mice | dogs | ... (x10000)
# 1 | 0  | 0  | 0  |  1   |  0   |  0   |  0   |   1  |  0  (x 10000)
# but that's not efficient, instead we "tokenize" and return the index numbers instead, e.g.
# "I like dogs" = [1, 5, 9]


# First try to load a pre-saved tokenizer (for performance) 
# If that doesn't work, you will need to re-make the tokenizer and embedding matrix
tokenizer_file <- "data/tokenizer"
if(file.exists(tokenizer_file)){
  tokenizer <- load_text_tokenizer(tokenizer_file)
} else{
  tokenizer <- text_tokenizer(num_words = max_vocab) %>%
               fit_text_tokenizer(dt$Text)
  save_text_tokenizer(tokenizer, tokenizer_file)
}

# You can get the word indexes like so:
tokenizer$index_word[1:10] %>%
  unlist() %>% 
  data.frame(word = ., index = names(.))

# Create equal length sequences of tokens
txt_sequences <- texts_to_sequences(tokenizer, dt$Text)
# Now pad/truncate so all are the same length. snip off super long comments at the beginning
text_data     <- pad_sequences(sequences  = txt_sequences, 
                               maxlen     = max_length)


#   2c Test/Train Split ========================================================

# Now to split our data into training and validation. Note that it's now trendy to call your
# validation set the "dev" set. In practice, you should make:
# Training set
# Validation (dev) set
# Test set
# Train the model on the training set and tweak it until it's working well on the validation set
# Once it looks good use the test set and make sure that it performas about as well on the test and validfation (dev) set! 

# Caret has a nice function to split our data to make sure that the responses are equally represented

set.seed(3364900) # fixes random number generation - for reproducability (within R, not CUDA!)
# Partition training and test data balanced by type (pro/con) &  use 80% in training
# Key to split is how many data points atre really needed for validation & testing
train_idx <- caret::createDataPartition(dt$target, p = .8, list = FALSE)

# Now split by balanced test/train indexes
x_train <- text_data[train_idx, ] %>% as.matrix() %>% unname()
x_test  <- text_data[-train_idx,] %>% as.matrix() %>% unname()

y_train <- dt[train_idx,  target] %>% as.matrix() %>% unname()
y_test  <- dt[-train_idx, target] %>% as.matrix() %>% unname()

# 3. Get Embedding =============================================================

# Embedding layers are vectors that contain information about the similarity/dissimilarity of words.
# An embedding layer contains knowledge of language structure and synonyms. 
# For example, that `emerald` is more closely related to `sapphire` than it is to `donkey`. 
# Creating language embeddings is a heavy task, and beyond our scope. 
# HOWEVER! I don't need to because I can take an open-sourced pre-trained language embedding model! 
# I'll use[a relatively small model pretrained on wiki news (word2vec). 
# 
# Extracting these embeddings and creating a weight matrix that can be fed into a new neural network.
# This is computationally intensive.
# It's best to do it in parallel and preferebly only once per combination of dictionary size and max sequence length. 

# 
# Check if this has already ran - no need to do it more than once! 
filename <- paste0("data/wiki_embedding_mx-300-", max_vocab, "-", max_length, ".RDS")
# print(filename)

if(file.exists(filename)){
  wiki_embedding_mx <- readRDS(filename)
} else {
  # Get Embeddings - give similarity between words used in English
  
  # 1: unzip embedding (>2GB)
  unzip("data/wiki-news-300d-1M.vec", )
  
  # 2: read unzipped object (will be in project dir)
  lines <- readLines("wiki-news-300d-1M.vec")
  # omit junk line 1
  lines <- lines[2:length(lines)]
  
  # 3: extract and map values to tokenizer
  # Container for embedding structure
  wiki_embeddings <- list()
  
  # This process takes a long time, so going to run in parallel
  require(pbapply)
  #24x faster...BUT no names..so effective 12x speedup
  wiki_embeddings <- pblapply(lines, 
                              function(line){
                                values     <- strsplit(line, " ")[[1]]
                                out        <- as.double(values[-1])
                                return(out)
                              }, 
                              cl = parallel::makeCluster(parallel::detectCores()))
  # So need to get and extract embedding names (unfortunately can't do all at once)
  embedding_names <-  pblapply(lines, 
                               function(line){
                                 values     <- strsplit(line, " ")[[1]]
                                 word       <- values[[1]]
                                 return(word)
                               }, 
                               cl = parallel::makeCluster(parallel::detectCores()))
  # It really pains me that I can't return to names index in a cluster apply 
  names(wiki_embeddings) <- embedding_names
  str(head(wiki_embeddings))
  
  # free some RAM
  rm(lines)
  rm(embedding_names)
  gc()
  
  # 4. Create our embedding matrix
  word_index         <- tokenizer$word_index 
  wiki_embedding_dim <- 300
  wiki_embedding_mx  <- array(0, c(max_vocab, wiki_embedding_dim))
  
  for (word in names(word_index)){
    index <- word_index[[word]]
    if (index < max_vocab){
      wiki_embedding_vec <- wiki_embeddings[[word]]
      if (!is.null(wiki_embedding_vec))
        wiki_embedding_mx[index+1,] <- wiki_embedding_vec # Words without an embedding are all zeros
    }
  }
  
  # 6. save and clean up
  #Save so you don't need to do that again
  saveRDS(wiki_embedding_mx, filename)
  rm(wiki_embeddings)
  gc()
  # remove unzipped file
  file.remove("wiki-news-300d-1M.vec")
}


# 4. Make model ================================================================

# 
# The neural network schematic is created using the Keras interface to the Tensorflow back end. 
# Once the network's topology is set, the layers are combined and then the weights from the 
# pre-trained embedding layer (containing a general understanding of the English language) are superimposed 
# onto the network's embedding layer and are set to be frozen (non-trainable). 
# 
# Note that as this is just for demonstration purposes, 
# the model structure has not undergone pruning or optimization.
# I'm certain that with tuning & pruning I could get even better results! 


embedding_dim <- 300

# Setup input layer. Using 16-bit ints since smaller vocab to save VRAM
input <- layer_input(
  shape = list(NULL),
  dtype = "int16",
  name  = "input"
)

# Model layers

# Embedding - will populate with weights later
embedding <- layer_embedding(object     = input,
                             input_dim  = max_vocab, 
                             output_dim = embedding_dim, 
                             name      = "embedding")

# Long Short Term Memory
# if you get an error with CUDNN, uncomment dropout = 0.25
lstm <- layer_lstm(object               = embedding, 
                   units                = max_length, 
                   activation           = "tanh",
                   recurrent_activation = "sigmoid",
                   use_bias             = TRUE,
                   return_sequences     = FALSE, 
                   dropout              = 0.25, 
                   recurrent_dropout    = 0.25, 
                   name                 = "lstm")

# Hidden Layers - using trusty tanh + lil2 norm. (can fall back to relu is speed is dragging)
# see how you can chain on more layers!
# Add dropout if it overfits. 
hidden <- lstm %>%
  layer_dense(units = max_length, 
              activation = "tanh", 
              name = "hidden",  
              kernel_regularizer = regularizer_l1_l2(l1 = 0.005, l2 = 0.005)) %>%
  layer_dense(units = 16,
              activation = "tanh",
              name = "hidden4",
  )

# Output - sigmoid for probabilities
predictions <- layer_dense(object     = hidden,
                           units      = 1, 
                           activation = "sigmoid",
                           name       = "predictions")


# Bring model together - predictions is a chain of embedding -> lstm -> hidden -> output
model <- keras_model(input, predictions)

# Set embedding layer to be wiki news weights
# Freeze the embedding weights to prevent overfitting
get_layer(model, name = "embedding") %>% 
  set_weights(list(wiki_embedding_mx)) %>% 
  freeze_weights()



# Compile
model %>% compile(
  optimizer = optimizer_adam(),
  loss      = "binary_crossentropy",
  metrics   = "binary_accuracy"
)


print(model)


# 5. Train Model ===============================================================

# Now to train the model just feed in the training input and output, the
# batch size, validation data (to see how it performs on data it isn't being optimized on), 
# the number of `epochs` (times to cycle through all training data elements), tell it to shuffle the 
# order in which it sees data every time, then tell it to keep logs for us to evaluate. 
# 
# NOTE - this will use a LOT of system resources, so assume that your computer will freeze and crash if you
# run the below code.

# create a checkpoint (save best version off model as it trains)
#tensorboard("logs/md_log") 
checkpoint_path <- "cp.ckpt"

# Checkpoint callback - to take the top validating model
cp_callback <- callback_model_checkpoint(
  filepath          = checkpoint_path,
  save_weights_only = FALSE,
  save_best_only    = TRUE,
  verbose           = 1
)

history <- fit(object          = model,
               x               = x_train,
               y               = y_train,
               batch_size      = 1024,
               validation_data = list(x_test, y_test),
               epochs          = 10,
               shuffle         = TRUE,
               view_metrics    = TRUE,
               callbacks       = list(cp_callback),
               verbose         = 1)

# Look at training results
print(history)


# 6. Evaluate ==================================================================
# Restore best model from checkpoint
top_model <- load_model_tf(checkpoint_path)
# Create predictions vs real (as factors for caret)
y_pred    <- predict(top_model, x_test)
y_pred    <- factor(round(y_pred))
y_real    <- factor(round(y_test))

conf_mx   <- caret::confusionMatrix(y_pred, y_real, positive="1")

keras::save_model_hdf5(top_model, "glassdoor_model.hdf5")

# always a good idea to save with the model
saveRDS(list(max_vocab  = max_vocab, 
             max_length = max_length,
             embed_dim  = embedding_dim,
             train_date = Sys.Date(),
             train_size = dim(x_train),
             train_hist = history,
             model_cmx  = conf_mx,
             notes      = "history is second shot 10 epocs with batch 128 (init had 13 with batch of 256)"),
        "glassdoor_specs.RDS")

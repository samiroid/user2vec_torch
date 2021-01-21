BASE_PATH="/Users/samir/Dev/projects/user2vec_v2/"
CORPUS=$BASE_PATH"/raw_data/sample.txt"
WORD_EMBEDDINGS=$BASE_PATH"/raw_data/word_embeddings.txt"
PKL_PATH=$BASE_PATH"/DATA/pkl/"
OUTPUT_PATH=$BASE_PATH"/DATA/out/"

# python user2vec/build.py -input $CORPUS -emb $WORD_EMBEDDINGS -output $PKL_PATH

python user2vec/train.py -input $PKL_PATH  -output $OUTPUT_PATH
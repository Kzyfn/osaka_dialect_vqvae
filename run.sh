python ./scripts/prepare_features.py ./data --use_phone_alignment --question_path="./data/questions_jp.hed"


python src/search_param.py -od 0810 -q True -nc 4 -nl 2 -zd 1 > vqvae.ou

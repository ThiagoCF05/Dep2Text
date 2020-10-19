export LC_ALL="en_US.UTF-8"

# Python project path
CHALLENGE=/home/tcastrof/Dep2Text

# europarl corpus path
#EUROPARL=/exp2/tcastrof/europarl/corpus
#EUROPARL_TOOLS=/home/tcastrof/workspace/europarl/tools

# moses path
MOSESDIR=/home/tcastrof/workspace/mosesdecoder

# kenlm path
KENLM=/home/tcastrof/workspace/kenlm/build/bin
# path to save training data for the language models
LM_TEXT=/roaming/tcastrof/europarl/clean
# path to save the languge models
LM_MODEL=/roaming/tcastrof/europarl/models

#mkdir $LM_TEXT
mkdir $LM_MODEL

# move to python project
cd $CHALLENGE

# parsing the data challenge
python parse2019.py
# training preordering model
python order.py
# find lexicon
python lexicon.py
# preordering and lexicalizing dependencies and making parallel with gold-standard
python mt.py

for lng in ar hi id ja ko ru zh
  do
    echo 'Language' $lng
    cd $CHALLENGE/data2019/mt/$lng
    cp test.de test.out
    $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $lng < test.out > test.detok.out

    echo 'Save results'
    python $CHALLENGE/utils.py $CHALLENGE/data2019/mt/$lng/test.out $CHALLENGE/data2019/mt/$lng/test.json $CHALLENGE/data2019/tok
    python $CHALLENGE/utils.py $CHALLENGE/data2019/mt/$lng/test.detok.out $CHALLENGE/data2019/mt/$lng/test.json $CHALLENGE/data2019/detok
  done

for lng in pt en es fr #ar hi id ja ko ru zh
  do
    echo 'Language' $lng
    cd $CHALLENGE/data2019/mt/$lng

   echo 'Preprocess the training data for the language model'
   python $CHALLENGE/europarl.py $EUROPARL/$lng $LM_TEXT/$lng
   perl $EUROPARL_TOOLS/tokenizer.perl -l $lng < $LM_TEXT/$lng > $LM_TEXT/$lng.tok
   perl $EUROPARL_TOOLS/split-sentences.perl -l $lng < $LM_TEXT/$lng.tok > $LM_TEXT/$lng.tok.pre

   echo 'Train language model with kenLM'
   $KENLM/lmplz -o 5 -T /tmp <train.en >$LM_MODEL/$lng.arpa
   $KENLM/build_binary $LM_MODEL/$lng.arpa $LM_MODEL/$lng.bin
   rm $LM_MODEL/$lng.arpa

   echo 'Normalize punctuation on the training set'
   perl $MOSESDIR/scripts/tokenizer/normalize-punctuation.perl < train.de > ntrain.de
   perl $MOSESDIR/scripts/tokenizer/normalize-punctuation.perl < train.en > ntrain.en

   echo 'Training moses'
   perl $MOSESDIR/scripts/training/train-model.perl \
       -root-dir . \
       --corpus ntrain \
       -mgiza \
       --max-phrase-length 9 \
       -external-bin-dir /home/tcastrof/workspace/mgiza \
       --f de --e en \
       --parallel \
       --distortion-limit 6 \
       --lm 0:5:$LM_MODEL/$lng.bin \
       -reordering phrase-msd-bidirectional-fe,hier-mslr-bidirectional-fe

   echo 'Tunning moses'
   perl $MOSESDIR/scripts/training/mert-moses.pl \
   	dev.de \
   	dev.en \
       $MOSESDIR/bin/moses \
       model/moses.ini \
       --mertdir $MOSESDIR/mert \
       --rootdir $MOSESDIR/scripts \
       --nbest 1000 \
       --decoder-flags '-threads 64 -v 0' \
       --batch-mira --return-best-dev \
       --batch-mira-args '-J 60'

    echo 'Decoding'
    $MOSESDIR/bin/moses -f mert-work/moses.ini -s 1000 < test.de > test.out
    $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $lng < test.out > test.detok.out

    echo 'Save results'
    python $CHALLENGE/utils.py $CHALLENGE/data2019/mt/$lng/test.out $CHALLENGE/data2019/mt/$lng/test.json $CHALLENGE/data2019/tok
    python $CHALLENGE/utils.py $CHALLENGE/data2019/mt/$lng/test.detok.out $CHALLENGE/data2019/mt/$lng/test.json $CHALLENGE/data2019/detok
    
    python contractions_pt.py
  done
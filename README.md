# caption_competition
source code of CMU-RUC entrances at various caption competitions

* best performer at video to text task in [TRECVID2018](https://www-nlpir.nist.gov/projects/tv2018/Tasks/vtt/)
* best performer at video to text task in [TRECVID2017](https://www-nlpir.nist.gov/projects/tv2017/Tasks/vtt/)
* best performer at [MSR vision to language challenge 2017](http://ms-multimedia-challenge.com/2017/challenge)
* best performer at [MSR vision to language challenge 2016](http://ms-multimedia-challenge.com/2016/challenge)

depends on my personal [tensorflow experiment framework](https://github.com/marshimarocj/tf_expr_framework)

## gen_model
caption generation models
* vevd.py: vanilla encoder (one full connect layer) and LSTM-cell rnn decoder, trained with cross-entropy loss
* vead.py: vanilla encoder (one convolutionallayer) and LSTM-cell rnn decoder with attention, trained with cross-entropy loss
* self_critique.py: vanilla encoder (one full connect layer) and LSTM-cell rnn decoder, trained with selfcritical loss
* diversity.py: vanilla encoder (one full connect layer) and vanilla rnn decoder, trained with diversity loss
* margin.py: vanilla encoder (one full connect layer) and vanilla rnn decoder, trained with contrastive-like loss

## decoder
* rnn.py: LSTM-cell rnn decoder
* att_rnn.py: LSTM-cell rnn decoder with attention

## rank_model
caption retrieval model, used in reranking the outputs from multiple caption generation models
* rnnve.py: bidirectional rnn encoder on caption and vanilla encoder (one full connect layer) of video feature
* rnnve_embed.py: bidirectional rnn encoder with pretrained word embeddings on caption and vanilla encoder (one full connect layer) of video feature
* ceve.py convolutional encoder on caption and vanilla encoder (one full connect layer) of video feature

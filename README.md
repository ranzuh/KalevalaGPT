# KalevalaGPT

Training a GPT-like transformer to generate the next characters based on previous ones. 

## Baseline Bigram language model

The Bigram language model predicts the next character based on just the previous character. Training the model for 5000 iterations scores a 2.462 cross-entropy loss on the validation set.

```zsh
% python bigram_model.py
...
Iter 4999 - Train loss: 2.471, Validation loss: 2.462

mi,
' leoikksys S:
sin i kaOäyAjuronäUHolKasitt oi Je oan
tutuenen ma:algLeni pä roi
katt,
sa:
naura!masta..
a,
KOhei, päri Ä: sejo;
"SKi:
kuoa kenpuujaa tivaaivi:
ran,
jHhän velennetatu' rsi,
koilunj
```

## GPT-like transformer

This transformer uses self-attention to predict the next character based on previous context. During training, it learns to attend to different parts of the context to better predict the next word. The architecture is similar to the decoder introduced in the seminal paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Training the model for 5000 iterations vastly improves over the Bigram model and scores a 1.479 cross-entropy loss on the validation set. The model is very tiny and the training takes less than five minutes on M1 Macbook Air CPU.  Training a more complex model on proper GPU will lead to a much better performance.

```zsh
% python gpt_model.py
...                                                                                             
Iter 4999 - Train loss: 1.284, Validation loss: 1.479

Päivän päivänä mieltä,
siihen suuren jälestä,
alusi kuulen luoja,
kaa'a karjan kantajata,
kaunoilla kantelevat!
Kullervo, Kalervon poika,
Siinä vanhan virukan,
entisillä aian ansatus
kuss' olet kussit käessäsi,
kulkea omentamahan,
koivu kuulematta,
kävinät kätösiä kämennyt.
"Äiti Lemminkäinen luoma,
```


## Acknowledgements

The source code here is heavily based on [Andrej Karpathy's](https://github.com/karpathy) YouTube lesson [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY). I tried to reproduce it from memory after watching the lesson but had to peak back and copy some parts.

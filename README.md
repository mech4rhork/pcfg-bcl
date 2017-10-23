# pcfg-bcl
Implementation of PCFG-BCL by Kewei Tu and Vasant Honavar \[1\].

\[1\] Tu, K., & Honavar, V. (2008, September). Unsupervised learning of probabilistic context-free grammar using iterative biclustering. In _ICGI_ (pp. 224-237). [pdf](http://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1241&context=cs_techreports)

## Usage
```
tugram.py learning_corpus generated_grammar
```

## File descriptions
* __tugram.py__ - Main script. Learns a PCFG (output) from a learning corpus (input).
* __pcfg_bcl.py__ - PCFG-BCL implementation.
* __grammars.py__ - Functions used to generate test corpora from PCFGs.
* __test.py__ - Tests from section 5 in the paper \[1\].
* __\*.txt__ - Test corpora.

## Performance evaluation
| Corpus\Score | Precision | Recall | F-score |
|:------------:|:---------:|:------:|:-------:|
| Baseline | 90.0 | 100 | 93.3 |
| Num-agr | 45.5 | 100 | 61.8 |
| Langley1 | 88.0 | 100 | 89.4 |
| Langley2 | 100 | 100 | 100 |

## Requirements
* Python 2.7+
* nltk
* numpy
* pandas
* coclust

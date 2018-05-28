# Dep2Text
Approach developed by the Tilburg University team to the shallow track of the [Multilingual Surface Realization Shared Task 2018 (SR18)](http://taln.upf.edu/pages/msr2018-ws/#sharedtask)

## Dependencies

To execute the code, you will need to have [NLTK](http://www.nltk.org/install.html), [Spacy](https://spacy.io/usage/) and [Moses](http://www.statmt.org/moses/?n=Development.GetStarted) installed. Make sure to follow the provided links to proper install them.

As an external resource, we also use [the Europarl Parallel Corpus](http://www.statmt.org/europarl/) to train the language models of our surface realization approach. To download the corpus and the tools to process it, you can execute the following command:

```
wget http://www.statmt.org/europarl/v7/europarl.tgz
wget http://www.statmt.org/europarl/v7/tools.tgz
```

## Approach

Once all te dependencies are proper installed, our approach can be executed by updating the paths in the `main.sh` script and run it with the following command:

`sh main.sh`

**Author:** Thiago Castro Ferreira
**Date:** May 28th, 2018

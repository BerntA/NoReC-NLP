ABOUT

The Norwegian Review Corpus (NoReC) was created for the purpose of training and
evaluating models for document-level sentiment analysis. This first release of
the corpus comprises 35,194 full-text reviews (approx. 15 million tokens) for a
range of different categories, extracted from eight different news sources:
Dagbladet, VG, Aftenposten, Bergens Tidende, Fædrelandsvennen, Stavanger
Aftenblad, DinSide.no and P3.no.  Each review is labeled with a manually
assigned score of 1–6, as provided by the rating of the original
author. Ratings and other metadata for all reviews are provided in a single
JSON file, while the text of each review is stored as a separate file (both
CoNLL-U and HTML), with the filename given by the review ID. The data set comes
with predefined splits for training, development and testing.

DOCUMENTATION

The full documentation of the resource is maintained online at the following
git page, which also includes several utility scripts:

https://github.com/ltgoslo/norec

For questions, please open an issue at the git page or send an email to
'sant-dev [at] ifi.uio.no'.


LICENCE

The data is distributed under a Creative Commons Attribution-NonCommercial
licence (CC BY-NC 4.0). The licence means that third parties can not
redistribute the original reviews for commercial purposes. Note however, that
machine learned models, extracted lexicons, word embeddings, and similar
resources that are created on the basis of NoReC are not considered to contain
the original data and so CAN BE FREELY USED ALSO FOR COMMERCIAL PURPOSES
despite the NC condition of the licence. Access the full license text here:

https://creativecommons.org/licenses/by-nc/4.0/


CITING

When using or referencing NoReC, please cite the following paper (arXiv
preprint):

NoReC: The Norwegian Review Corpus
Erik Velldal, Lilja Øvrelid, Eivind Alexander Bergem, Cathrine Stadsnes, Samia Touileb, Fredrik Jørgensen
2017
https://arxiv.org/abs/1710.05370

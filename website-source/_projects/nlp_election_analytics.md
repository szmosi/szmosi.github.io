---
layout: page
title: "NLP Election Analytics: Classical vs Transformer Models"
description: "An MSc group project comparing sentiment analysis and topic modelling methods on 2024 US election-related Twitter/X discussions."
img: assets/img/projects/nlp_election/bertopic_top_topics.png
importance: 1
category: "Data Science"
github: https://github.com/szmosi/nlp-election-analytics
---

## Overview

How do traditional NLP approaches and transformer-based models differ when analysing large volumes of short, informal social-media posts? This **collaborative MSc coursework project**, completed in 2025, compared sentiment analysis and topic discovery methods using a publicly documented dataset of Twitter/X posts about the **2024 US presidential election**.

The analysis explored **model-generated sentiment**, frequently discussed themes, and computational trade-offs. This was an exploratory methodological comparison, **not** an election forecast or a representative poll of US voters.

**Methods:** Python · Pandas · spaCy · NLTK · Gensim LDA · VADER · TextBlob · SentenceTransformers · UMAP · KMeans · BERTopic · Twitter-RoBERTa

**Source code:** [View the cleaned coursework notebooks on GitHub](https://github.com/szmosi/nlp-election-analytics).

## Analysis workflow

The original study used two parallel routes after dataset preparation. The classical route paired **LDA** topic modelling with **VADER/TextBlob** sentiment scoring; the transformer route used **BERTopic** for themes and a pretrained **Twitter-RoBERTa** classifier for sentiment.

![Original two-path NLP analysis workflow]({{ '/assets/img/projects/nlp_election/workflow_original.png' | relative_url }})
*Original coursework workflow diagram. The two routes use different text representations and preprocessing assumptions.*

## Dataset and scale

The original research used the [USC X 24 US Election dataset](https://github.com/sinking8/usc-x-24-us-election), described in [Balasubramanian et al. (2024)](https://arxiv.org/abs/2411.00376). The coursework report describes approximately **47 million collected posts** dated May–November 2024. After filtering invalid records and focusing on English, the analysis reports **36,784,993 posts**. Due to resource constraints, it used samples of **9 million posts for the classical route** and **3 million for the transformer route**.

The project uses historical social-media data rather than a probability sample of voters. Tweet activity, account behavior, selection rules, and classifier limitations all affect the interpretation. The dataset is **not redistributed** in the code repository; consult the data provider's repository for access and reuse conditions.

## Path A — classical NLP

For topic discovery, the project applied **Gensim LDA** to lemmatized Bag-of-Words representations, selecting **10 topics** after exploratory topic-coherence checks. The report records a **C_v coherence score of 0.389**, with some overlap between topic clusters. Topic names were assigned manually for interpretation.

For sentiment, **VADER** and **TextBlob** scored the posts using lexicon-based methods. The original results showed that the two methods produced visibly different score distributions, illustrating how method choice influences downstream interpretation.

![Histogram of VADER compound sentiment scores from the original classical NLP notebook]({{ '/assets/img/projects/nlp_election/vader_sentiment_distribution.png' | relative_url }})
*VADER compound-score distribution, exported from the original coursework notebook. Scores are model outputs, not human-verified sentiment labels.*

![Weekly VADER sentiment by selected LDA topics]({{ '/assets/img/projects/nlp_election/lda_topic_sentiment_over_time.png' | relative_url }})
*Illustrative topic-level trends from the original classical-analysis notebook. Interpret short-term fluctuations cautiously.*

## Path B — transformer-based NLP

The topic-modelling route used **all-MiniLM-L6-v2** sentence embeddings, **UMAP** dimensionality reduction and **KMeans** clustering within a **BERTopic** workflow. The report describes more than **2,000 initial topic clusters**, many small or overlapping, and focuses its discussion on prevalent topics. The reported **C_v coherence score was 0.378**. Directly comparing this number with LDA's score is insufficient to establish which model was better: the topic counts and modelling procedures differ.

For sentiment, the project applied a **pretrained Twitter-RoBERTa** classifier in batches, rather than fine-tuning a model on newly labeled election posts.

![Top BERTopic clusters by tweet count]({{ '/assets/img/projects/nlp_election/bertopic_top_topics.png' | relative_url }})
*Most frequent BERTopic clusters in the saved notebook output. Cluster names are generated labels and require human interpretation.*

![Counts of RoBERTa sentiment labels]({{ '/assets/img/projects/nlp_election/roberta_sentiment_distribution.png' | relative_url }})
*Distribution of the pretrained model's negative, neutral and positive classifications on the analysed sample.*

![Topic-level proportions of RoBERTa sentiment classes]({{ '/assets/img/projects/nlp_election/bertopic_sentiment_heatmap.png' | relative_url }})
*Topic-level sentiment proportions from the saved BERTopic notebook output. Model classifications are not ground-truth observations.*

## Findings and trade-offs

- **Topic granularity:** The fixed-topic LDA baseline generated broad, manually interpreted themes. BERTopic identified a larger collection of more specific clusters, but these also required consolidation and interpretation.
- **Sentiment disagreement:** VADER, TextBlob and RoBERTa did not yield the same sentiment distributions. Such differences highlight the importance of model choice and validation rather than conclusively proving any model correct.
- **Compute requirements:** Lexicon-based scoring is relatively lightweight. Embedding generation and transformer inference are more computationally demanding and benefited from sampling and GPU-enabled processing.
- **Reproducibility:** The public notebooks document the original workflow, but reproducing the historical large-scale outputs requires the original data, intermediary artifacts, compatible dependencies and substantial compute resources. The cleaned notebooks have not been claimed to reproduce the original results end to end.

## Limitations

The analysis did not use a manually labeled election-specific test set to calculate comparable accuracy metrics. Social-media discussions are not representative measures of voter attitudes; sarcasm, language, topic labelling and sampling can influence classifications. The work was completed collaboratively as an **MSc course group assignment**, and this page describes group-level results without attributing the entire implementation to any one team member.

## Data and project links

- [Public code repository](https://github.com/szmosi/nlp-election-analytics)
- [Original dataset repository](https://github.com/sinking8/usc-x-24-us-election)
- [Dataset paper: Balasubramanian et al. (2024)](https://arxiv.org/abs/2411.00376)

*Visualizations on this page were extracted from saved outputs of the original coursework notebooks; they were not regenerated from the cleaned public code.*

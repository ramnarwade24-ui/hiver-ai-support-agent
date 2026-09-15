# Hiver SDE Intern - AI Customer Support Agent

## Overview

This project builds an AI support agent for AmazonHelp using the Customer Support on Twitter dataset.

The agent:
1. Classifies customer messages into support intents.
2. Retrieves historically similar AmazonHelp cases.
3. Drafts a support response.
4. Decides whether to auto-handle or escalate.

## Dataset

Brand selected: AmazonHelp.

After cleaning and deduplication, the historical retrieval dataset contains approximately 154,910 customer-to-brand response pairs.

## Intent Classification

The project defines 14 support intents covering delivery, orders, returns, payments, account access, Prime, products, security, complaints, and other requests.

## Golden Evaluation Set

A 200-example coverage-oriented golden set was used.

The set was sampled to increase coverage of diverse issue patterns rather than estimate the natural class distribution.

File:

amazon_golden_200_labeling.csv

Important limitation: prime_video has no examples in the final golden set, while fraud_or_security has only two examples.

## Results

Majority-class baseline:

- Accuracy: 20.5%
- Macro-F1: 2.6%

TF-IDF + Logistic Regression:

- OOF Accuracy: 43.0%
- OOF Macro-F1: 34.8%
- Correct: 86 / 200

The evaluation uses out-of-fold predictions to avoid training/evaluation leakage.

## Historical Retrieval

Historical customer messages are represented using TF-IDF word unigrams and bigrams.

Cosine similarity retrieves similar historical customer cases.

Very short, generic, and unsuitable historical responses are filtered before being used as evidence.

## Response Generation

The current prototype uses intent-specific response templates combined with historical retrieval evidence.

## Escalation

High-risk intents such as fraud_or_security and customer_service_complaint are escalated.

The system also escalates when classifier confidence or historical retrieval evidence is insufficient.

The current conservative policy resulted in:

- Auto-handled: 0 / 200
- Escalated: 200 / 200

This is treated as a safety finding rather than artificially lowering thresholds to increase automation.

## Failure Modes

Observed failure modes include:

1. Delivery-delay messages confused with Prime-related language.
2. Delivery-not-received confused with delivery-delayed.
3. Payment/billing confused with other requests.
4. Return/refund confused with customer-service complaints.
5. High retrieval similarity can still return an unsuitable historical response.

## Artifacts

- amazon_intent_classifier.joblib
- amazon_retrieval_vectorizer.joblib
- amazon_retrieval_matrix.joblib
- amazon_retrieval_cases.pkl
- amazon_golden_200_labeling.csv

## Limitations

The golden set contains only 200 examples and is coverage-oriented rather than naturally distributed.

Some intents have limited support.

Classifier confidence is not treated as calibrated probability.

The current system is intentionally conservative about autonomous handling.

## Reproduction

Install dependencies from requirements.txt and run the supplied Colab/Jupyter notebook.

The notebook contains data preparation, intent discovery, evaluation, retrieval, response generation, and escalation logic.

## Headline Number Caveat

The 43.0% accuracy and 34.8% macro-F1 should not be interpreted as production performance. They are measured on a small, coverage-oriented 200-example golden set with uneven class support.

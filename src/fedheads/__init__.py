"""Classifier heads in federated sentiment analysis.

The training engine (federated loop, LoRA setup, data loading, seeding, checkpoints) was
copied from the script-heterogeneity repo (`Script-Heterogeneity-in-Federated-Learning`,
tag `fedblacx-before-move`, commit eadd3cf) and cut down to what this study needs: one
language per client, one view of the text, and FedAvg, local-only and centralized training.
"""

SOURCE_TAG = "fedblacx-before-move"
SOURCE_COMMIT = "eadd3cf"

# UltraPose

This project aims to generate high-fidelity and behaviorally diverse ultrasonic data for augmenting datasets in identification, and is particularly effective when only a few real samples are available.

requirement.txt: required environments.

gen_cond.py: generate speed and position shift embeddings (conditions) matrix for two instances.

dataset.py: generate dataset from files.

model.py: architecture of conditional Unet.

train.py: run to train UltraPose model.

utils.py: save/load model function.
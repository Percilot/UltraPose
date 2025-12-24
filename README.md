# UltraPose

This project aims to generate high-fidelity ultrasonic data with rich kinetic diversity to support few-shot ultrasonic sensing.

requirement.txt: required environments.

gen_cond.py: generate speed and position shift embeddings (conditions) matrix for two instances.

dataset.py: generate dataset from files.

model.py: architecture of conditional Unet.

train.py: run to train UltraPose model.

utils.py: save/load model function.
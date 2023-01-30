## ISS Image City Classifier
- This project was written for my MSc Computer Science final project. 
It uses PyTorch to classify cities within nocturnal (nighttime) DSLR imagery from
the International Space Station.

- Data from the ISS archive can be downloaded automatically using the `data_acquisition/data_downloader.py`
script, and preprocessed using the `preprocessing_pipeline/` module.

- To train the CNN model and evaluate it, use the scripts in the `transfer_learning/` directory. 
To train the model, run `model_training.py`, and to evaluate the model, run`model_evaluation.py`.

- The training parameters can be changed using the `transfer_learning/training_consfig.toml` file.

- The libraries required to run this project can be installed using `pip install -r requirements.txt`.
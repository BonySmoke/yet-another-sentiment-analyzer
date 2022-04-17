# Yet Another Sentiment Analyser

## Purpose
The goal of this ML analyser is to provide a system that can solve the following problems:
- Train your own sentiment analysis model.
- Provide an API endpoint to integrate the system with third-part services.
- Have a web-interface to interact with the analyser online.

## About
The analyser is built using the [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), in particular game reviews.

Multiple models have been built to server different needs, including the binary model and fine-grained ones.

The analyser supports at least 2 languages: Ukrainian and English but can be easily trained to work with other languages is it's built using statistical methods.

The overall performance of the system doesn't correspond to the production-ready expectations (mainly due to a lack of computational power during the model training) but may server as a foundation for the creation of more complex models.

## Training a model
To train a model, examine code in file ./examples/train_model.py

## Accuracy
Here are a couple of confusion matrices for models built using linear regression.
- Three classes (Positive, Neutral, Negative)<br/>
  Accuracy: ~**71%**<br/>
![Cool!](./assets/linear_three.png?raw=true "Three classes (Positive, Neutral, Negative)")<br/>

- Binary (Positive, Negative)<br/>
  Acuracy: ~**92%**<br/>
![Cool!](./assets/linear_binary.png?raw=true "Three classes (Positive, Neutral, Negative)")<br/>

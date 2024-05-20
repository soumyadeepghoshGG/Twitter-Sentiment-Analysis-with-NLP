# Twitter Sentiment Analysis using NLP


## Installation

To run the project, you'll need to have the following Python packages installed. You can install them using the following command:

```bash
pip install python-dateutil numpy pandas matplotlib seaborn scikit-learn nltk wordcloud spacy emot emoji
```

## Natural Language Processing
<center><img src=https://cdn.hashnode.com/res/hashnode/image/upload/v1623129057923/pMTBDSw6l.png?w=1600&h=840&fit=crop&crop=entropy&auto=compress,format&format=webp></center><br>
Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language in a way that is both meaningful and useful. Essentially, it's about teaching computers to comprehend and interact with human language, whether it's written or spoken. NLP techniques allow computers to perform tasks such as language translation, sentiment analysis, speech recognition, and text summarization. By leveraging algorithms and linguistic rules, NLP systems can extract meaning from text data, enabling a wide range of applications across various domains like healthcare, finance, customer service, and more.

## Sentiment Analysis
Sentiment Analysis, also known as opinion mining, is a subfield of NLP that involves analyzing and understanding the sentiment or emotion expressed in text data. The goal of sentiment analysis is to determine the attitude, opinion, or emotion conveyed by a piece of text, whether it's positive, negative, or neutral. By employing machine learning algorithms and linguistic analysis techniques, sentiment analysis systems can automatically classify text as expressing positive, negative, or neutral sentiment. This technology finds widespread use in social media monitoring, customer feedback analysis, brand reputation management, market research, and other applications where understanding public opinion is crucial. Essentially, sentiment analysis helps businesses and organizations make data-driven decisions by gauging the sentiment of their audience or customers.

## About Dataset

### Context
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

### Content
It contains the following 6 fields:

1. **target:** the polarity of the tweet (0 = negative, 4 = positive)
2. **ids:** The id of the tweet (2087)
3. **date:** the date of the tweet (Sat May 16 23:58:44 UTC 2009)
4. **flag:** The query (lyx). If there is no query, then this value is NO_QUERY.
5. **user:** the user that tweeted (robotickilldozr)
6. **text:** the text of the tweet (Lyx is cool)

### Some Functions Used
```bash
# Preprocessing text data
text = "Hello, happy learning!"
processed_text = nlp_preprocessor(text)

# For more information on how it works
print(help(nlp_preprocessor))

# Predicting sentiment
detect_sentiment(text, preprocessor=nlp_preprocessor, model=model, vectorizer=vectorizer)
# Assuming you have a classification model named 'model' and a vectorizer named 'vectorizer'
# You can load a trained model and vectorizer from pickled files in the 'models' folder

# For more information on how it works
print(help(detect_sentiment))
```

### Algorithms Implemented
- Logistic Regression
- Bernoulli Naive Bayes Classifier
- Linear SVC [Support Vector Machine]

## Acknowledging Model Limitations
While our sentiment analysis model strives for accuracy, it's essential to acknowledge that it may not always achieve 100% precision. Like any predictive system, there are instances where the model might misclassify or inaccurately predict sentiment. This could occur due to various factors, such as ambiguous language, sarcasm, or the presence of uncommon expressions or context that the model hasn't encountered during training. Additionally, sentiment analysis is inherently subjective, and interpretations of sentiment can vary among individuals. Therefore, it's crucial to interpret the model's predictions with a degree of caution and to conduct error analysis to understand the reasons behind incorrect classifications. By identifying patterns in misclassifications, we can iteratively refine and improve the model's performance, ultimately enhancing its accuracy and reliability over time.

## Future Work
1. **Fine-tuning and Hyperparameter Optimization**: A key area for future work can be fine-tuning existing models and optimizing their hyperparameters to achieve better performance. By systematically adjusting model parameters and optimization strategies, we can enhance the accuracy, robustness, and generalization capabilities of sentiment analysis systems.

2. **Employing Deep Learning Models, Including LSTM**: Another promising direction is the utilization of deep learning architectures, such as recurrent neural networks (RNNs), convolutional neural networks (CNNs), and transformer-based models like BERT and GPT, for sentiment analysis. These models have demonstrated remarkable success in various NLP tasks, including sentiment analysis. By leveraging these sophisticated neural network architectures, sentiment analysis systems can achieve a more nuanced understanding and representation of textual data, capturing complex patterns and context dependencies inherent in human language. In particular, implementing LSTM (Long Short-Term Memory) networks can be beneficial for capturing long-term dependencies in sequential data, making them suitable for sentiment analysis tasks involving text sequences.

3. **Collecting More Data**: Continuously gathering more data, particularly the latest ones, is essential for improving the accuracy and relevance of sentiment analysis models. As language evolves over time and new expressions, slang, and topics emerge, training data must reflect these changes to ensure that sentiment analysis systems remain effective and up-to-date. By collecting and incorporating the latest data from various sources such as social media platforms, news websites, and online forums, researchers can enhance the model's ability to capture contemporary language usage and sentiment trends. Additionally, employing techniques such as active learning and data augmentation can help in efficiently acquiring and augmenting labeled data for training sentiment analysis models, thereby improving their performance on real-world tasks.

## Reference
Citation: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](https://github.com/soumyadeepghoshGG/Twitter-Sentiment-Analysis-with-NLP/blob/main/License.txt) file for details.

## Contact
For questions or issues, please contact me (Soumyadeep Ghosh) via mail: soumyadeepghosh57@gmail.com
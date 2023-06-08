import nltk 
import nltk.corpus
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import networkx as nx
import numpy as np
import PyPDF2


def read_article(file_name):
    if file_name.endswith(".pdf"):
        return read_pdf(file_name)
    else:
        return read_txt(file_name)

def read_pdf(file_name):
    with open(file_name, 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        text = ""
        for page in pdfReader.pages:
            text += page.extract_text()
        
    sentences = text.replace("\n", " ").split(". ")
    return [s.replace("[^a-zA-Z]", " ").split(" ") for s in sentences if s]



def read_txt(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()
    article = filedata.split(". ")
    sentences = [s.replace("[^a-zA-Z]", " ").split(" ") for s in article if s]
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]

    # If both sentences are empty, return 0
    if not sent1 and not sent2:
        return 0

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        vector1[all_words.index(w)] += 1
    
    # build the vector for the second sentence
    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: # ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words("english")
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = read_article(file_name)

    # Step 2 - Generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print("Indexes of top ranked_sentence order are: ", ranked_sentence)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    print("Summarize Text: \n", ". ".join(summarize_text))


if __name__ == "__main__":
    generate_summary("improving_language_models_retrieving_trillions_tokens.pdf", 2)

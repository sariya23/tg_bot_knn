import joblib

import re
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
    NewsNERTagger,
    MorphVocab
)


MODEL_KNN = joblib.load('model/model_knn.pkl')
VECTORIZER = joblib.load('model/vectorizer.pkl')
FOLDER_NAMES_LIST = joblib.load('meta_data/folders_names_list.pkl')
STOP_WORDS_LIST = joblib.load('meta_data/stop_words_list.pkl')


class ClassifyText:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def classify_text(self) -> str:
        cleaned_text = self.__read_from_file()
        text_lemmatized = self.__prepare_text(cleaned_text)
        text_vector = VECTORIZER.transform([text_lemmatized])
        pred_class_idx = MODEL_KNN.predict(text_vector)[0]
        return FOLDER_NAMES_LIST[pred_class_idx]

    def __read_from_file(self):
        with open(self.file_path, 'r', encoding='utf8') as f:
            contents_list = f.readlines()
        text = ""
        for i in contents_list:
            text += i
            text = text.replace('.', '')
            text = text.replace(',', '')
            text = text.replace(' — ', ' ')
            text = text.replace(' - ', ' ')

            text = text.replace('\n', ' ')
            text = text.replace(' ', ' ')
        return text

    def __prepare_text(self, text_initial: str) -> str:
        text_initial_words_list = re.findall(r'\b\S+\b', text_initial)
        text_initial_words_count = len(text_initial_words_list)
        text_prepared_1 = []
        text_initial_words_list_without_stop_words = []
        for i in range(text_initial_words_count):
            word = text_initial_words_list[i].lower()
            if word not in STOP_WORDS_LIST:
                text_initial_words_list_without_stop_words.append(text_initial_words_list[i])
        text_prepared_1_list = text_initial_words_list_without_stop_words[:]
        text_prepared_1 = ' '.join(word for word in text_prepared_1_list)

        text_prepared_2 = []
        text_prepared_2_1 = []
        segmenter = Segmenter()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        syntax_parser = NewsSyntaxParser(emb)
        doc = Doc(text_initial)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        ner_tagger = NewsNERTagger(emb)
        doc.tag_ner(ner_tagger)
        morph_vocab = MorphVocab()
        named_entities_list = []
        named_entities_count = len(doc.spans)
        for i in range(named_entities_count):
            named_entities_list.append(doc.spans[i].text)
        text1 = text_initial
        for i in range(named_entities_count):
            text1 = text1.replace(named_entities_list[i], '')
        text1 = text1.replace(' ', ' ')
        text1 = text1.replace(' ', ' ')
        text_prepared_2_1 = text1[:]

        text_prepared_2_1_words_list = re.findall(r'\b\S+\b', text_prepared_2_1)
        text_prepared_2_1_words_count = len(text_prepared_2_1_words_list)
        text_words_list_without_numbers = []

        for i in range(text_prepared_2_1_words_count):
            word = text_prepared_2_1_words_list[i]
            if not word.isdigit():
                text_words_list_without_numbers.append(word)
        text_prepared_2_list = text_words_list_without_numbers[:]
        text_prepared_2 = ' '.join(word for word in text_prepared_2_list)

        text_prepared_3 = []

        doc = Doc(text_prepared_2)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        dict_word_and_lemma = {j.text: j.lemma for j in doc.tokens}
        text_prepared_3_list = list(dict_word_and_lemma.values())
        text_prepared_3 = ' '.join(word for word in text_prepared_3_list)

        return text_prepared_3


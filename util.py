import string
import re


def preprocess(text):
  text = ''.join(ch for ch in text if ch not in string.punctuation)
  text = text.lower()
  text = re.sub(r'\d','',text)
  text = re.sub(r'\s+',' ',text)
  text = text.strip()
  return text


def unique_sentences(eng_sen, hin_sen):
  eng_unique = set()
  eng_sentences = []
  hin_sentences = []
  for i in range(len(eng_sen)):
    if eng_sen[i] not in eng_unique:
      eng_unique.add(eng_sen[i])
      eng_sentences.append(eng_sen[i])
      hin_sentences.append(hin_sen[i])
  
  return eng_sentences, hin_sentences

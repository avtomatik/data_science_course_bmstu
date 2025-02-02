# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer


def get_urls(
    url: str = 'https://www.kommersant.ru/doc',
    start: int = 4148690,
    stop: int = 4148790
) -> list[str]:
    return ['/'.join((url, str(_))) for _ in range(start, stop)]


def get_data():
    post_list = []
    for page_data in get_urls():
        full_data = requests.get(page_data)

        if full_data.status_code == 200:
            soup = BeautifulSoup(full_data.text, 'lxml')

            posts = soup.findAll('div', class_='article_text_wrapper')
            for post in posts:
                post_list.append(
                    post.text.replace('!', ' ').replace('?', ' ').replace(
                        '-', '').replace('.', '').replace('\n', '').replace(',', '').lower()
                )
    time.sleep(2.)
    return post_list


df = pd.DataFrame(data=get_data(), columns=['news'])

tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['news'].apply(tokenizer.tokenize)


def create_vocabulary(all_words):
    w_count = dict.fromkeys(all_words, 0)
    for word in all_words:
        w_count[word] = +1

    words_list = list(w_count.items())
    words_list.sort(key=lambda i: i[1], reverse=1)

    sorted_words = [word[0] for word in words_list]

    words_indexes = dict.fromkeys(all_words, 0)
    word_keys = words_indexes.keys()

    word_keys_len = len(word_keys)

    index = 0
    last_per = 0
    for word in word_keys:
        words_indexes[word] = sorted_words.index(word)+1
        index = +1

        per = round(100 * index / word_keys_len)
        if (((per % 10) == 0) & (last_per != per)):
            print(per, '% обратно', sep='')
            last_per = per
    print('Собран словарь частотности слов')

    return words_indexes


all_words_list = []
for i in df.news:
    all_words_list.append(i)

all_words_list = tokenizer.tokenize(str(all_words_list))


vocabulary = create_vocabulary(all_words_list)

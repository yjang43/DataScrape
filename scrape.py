#!/usr/bin/env python
"""
The script scrapes news articles from the domain specified in "newspaper.json".
Then saves articles under "article" directory
"""
import os
import json
from urllib.parse import urljoin
import newspaper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_DIR = os.path.join(BASE_DIR, 'articles/')

LIMIT = 300    # limit the number of article to retrieve from each news domain


with open('newspapers.json') as f:
    companies = json.load(f)

file_num = 0
for company, value in companies.items():
    print(company, value['link'])
    print(type(value['link']))
    paper = newspaper.build(value['link'], language='en', memoize_articles=False)
    print(paper.size())
    count = 0
    for article in paper.articles:
        if count == LIMIT:
            break
        try:            # check for any cases that fails to download the article
            article.download()
            article.parse()
        except newspaper.article.ArticleException:
            with open('log.txt', 'a') as f:
                f.write("article {} failed to download".format(article.url))
            continue
        print(article.url)
        print(article.text)
        if len(article.text) < 200:     # skip meaningless articles. less than 40 words articles
            continue
        article_json = {
            'url': article.url,
            'text': article.text
        }
        print(urljoin(ARTICLES_DIR, '{:05d}.json'.format(file_num)))
        with open(urljoin(ARTICLES_DIR, '{:05d}.json'.format(file_num)), 'w') as f:
            json.dump(article_json, f)

        count += 1
        file_num += 1



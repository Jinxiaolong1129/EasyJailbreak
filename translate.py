from typing import List
import pandas as pd
import requests

class Translate():
    """
    Translate is a class for translating the query to another language.
    """
    def __init__(self, language='en'):
        self.language = language
        languages_supported = {
            'en': 'English',
            'zh-CN': 'Chinese',
            'it': 'Italian',
            'vi': 'Vietnamese',
            'ar': 'Arabic',
            'ko': 'Korean',
            'th': 'Thai',
            'bn': 'Bengali',
            'sw': 'Swahili',
            'jv': 'Javanese'
        }
        if self.language in languages_supported:
            self.lang = languages_supported[self.language]
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def translate(self, text, src_lang='auto'):
        """
        translate the text to another language
        """
        googleapis_url = 'https://translate.googleapis.com/translate_a/single'
        url = '%s?client=gtx&sl=%s&tl=%s&dt=t&q=%s' % (googleapis_url, src_lang, self.language, text)
        data = requests.get(url).json()
        res = ''.join([s[0] for s in data[0]])
        return res
    
    
    
def translate_queries(file_path, languages):
    df = pd.read_csv(file_path)
    translators = {lang: Translate(lang) for lang in languages}
    total_queries = len(df)
    
    print("Starting translations...")
    for lang, translator in translators.items():
        print(f"Translating to {lang}...")
        translated_queries = []
        for i, query in enumerate(df['query']):
            translated_query = translator.translate(query)
            translated_queries.append(translated_query)
            print(f"Translated {i + 1}/{total_queries} queries to {lang}.")
        df[lang] = translated_queries
        df.to_csv("data/TDC_data_translations.csv", index=False)

    print("All translations completed.")
    df.to_csv("data/TDC_data_translations.csv", index=False)

    
# languages_to_translate = ['zh-CN', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw']
languages_to_translate = ['jv']
translate_queries('data/TDC_data_translations.csv', languages_to_translate)

    


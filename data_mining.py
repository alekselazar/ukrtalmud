import requests
from bs4 import BeautifulSoup as bs
import os
import numpy as np
from googletrans import Translator
from tokenizers import BertWordPieceTokenizer

def main():
    trans = Translator()

    pages = 5406
    current_page = 1

    eng_url = 'https://www.daf-yomi.com/Dafyomi_Page.aspx?id={0}&vt=6&fs=0'
    original_url = 'https://www.daf-yomi.com/Dafyomi_Page.aspx?id={0}&vt=5&fs=0'

    #Google Translate is limited to 5000 chars, so for original and english versions we want to
    #save all the pages, but for data set for the first training for ua translation, so we will make
    #two vocs for original an english

    original_voc = []
    eng_voc = []
    original_voc_success = []
    eng_voc_success = []
    ru_voc = []
    ua_voc = []
    wrong_pages = []

    while current_page <= pages:
        original_res = requests.get(original_url.format(current_page))
        original_text = bs(original_res.text).select("#ContentPlaceHolderMain_divTextWrapper")[0].find_all(['b','strong'])
        original_text = [i.text for i in original_text]
        original_text = ' '.join(original_text)
        original_voc.append(original_text)

        eng_res = requests.get(eng_url.format(current_page))
        eng_text = bs(eng_res.text).select("#ContentPlaceHolderMain_divTextWrapper")[0].find_all(['b','strong'])
        eng_text = [i.text for i in eng_text]
        eng_text = ' '.join(eng_text)
        print(len(eng_text))
        eng_voc.append(eng_text)
        #Russian translation from english with talmud terms is better, so first we translate to russian
        try:
            ru_ver = trans.translate(eng_text, dest='ru')
            ua_ver = trans.translate(ru_ver.text, dest='uk')
            eng_voc_success.append(eng_text)
            original_voc_success.append(original_text)
            ru_voc.append(ru_ver.text)
            ua_voc.append(ua_ver.text)
            print(f'Page {current_page} is comleete')
        except:
            wrong_pages.append(current_page)
            print(f'Page {current_page} is wrong - for manual translation')
        current_page += 1

    original_path = os.path.join('data_set', 'aramaic')
    eng_path = os.path.join('data_set', 'english')
    original_success_path = os.path.join('data_set', 'aramaic-success')
    eng_success_path = os.path.join('data_set', 'english_success')
    ru_path = os.path.join('data_set', 'russian')
    ua_path = os.path.join('data_set', 'ukrainian')

    os.mkdir(original_path)
    os.mkdir(eng_path)
    os.mkdir(original_success_path)
    os.mkdir(eng_success_path)
    os.mkdir(ru_path)
    os.mkdir(ua_path)

    for i in range(len(original_voc)):
        with open(os.path.join(original_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(original_voc[i])

    for i in range(len(eng_voc)):
        with open(os.path.join(eng_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(eng_voc[i])

    for i in range(len(original_voc_success)):
        with open(os.path.join(original_success_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(original_voc_success[i])

    for i in range(len(eng_voc_success)):
        with open(os.path.join(eng_success_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(eng_voc_success[i])

    for i in range(len(ru_voc)):
        with open(os.path.join(ru_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(ru_voc[i])

    for i in range(len(ua_voc)):
        with open(os.path.join(ua_path, f'{i+1}.txt'), 'w', encoding='utf8') as file:
            file.write(ua_voc[i])

    for path in [original_path, eng_path, original_success_path, eng_success_path, ru_path, ua_path]:
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False
        )

        tokenizer.train(files=[os.path.join(path, f) for f in os.listdir(path)], vocab_size=30000, min_frequency=2,
                        limit_alphabet=1000, wordpieces_prefix='##',
                        special_tokens=[
                            '[START]', '[END]', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

        tokenizer.save_model(path, 'tokens')

    with open(os.path.join('data_set', 'wrong_pages.txt'), 'w', encoding='utf8') as file:
        w = np.asarray(wrong_pages, dtype='str')
        w = ','.join(w)
        file.write(w)


if __name__=='__main__':
    main()


# Talmud Ukrainization
## Idea 
I strongly support my motherland, Ukaraine, in its struggle for freedom and independence during russian brutal invation.
In former Soviet Union Jewish were able to study Talmud only by English or russian translation. I think, Ukraininan Jewish community deserve to study Talmud on their native language.
## Data 
I collected data as original talmud text and english translation from https://www.daf-yomi.com/ with simple data scrappindg using `requests` and `BeautifulSoup`. After that, with `googletrans` library, english version was translated to russian and Ukrainian. Now, data is been manually cleaned.
## Neural net
After data preparation, it will be used to train Encoder-Decoder `Transformer` model with `MultiHeadAttention`.
## TODO
Now data is been manualy prepered. After that I will train the `model`
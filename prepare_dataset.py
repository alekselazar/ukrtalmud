from tokenizers import BertWordPieceTokenizer as Tokenizer
from tensorflow.data import Dataset

class PrepareDataset:
    def __init__(self, encoder_voc, decoder_voc, **kwargs):
        super(PrepareDataset, self).__init__(**kwargs)
        self.encoder_voc = encoder_voc
        self.decoder_voc = decoder_voc
        self.n_sentences = len(self.encoder_voc)
        assert self.n_sentences == len(decoder_voc)
        self.train_split = 0.9
    
    def create_tokenizer(self, voc):
        tokenizer = Tokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False
        )
        tokenizer.train_from_iterator(iterator=voc, vocab_size=30000, min_frequency=2,
                        limit_alphabet=1000, wordpieces_prefix='##',
                        special_tokens=[
                            '[START]', '[END]', '[PAD]', '[UNK]'])

        return tokenizer
    
    def find_seq_length(self):
        return max(len(seq.split()) for seq in self.encoder_voc)

    def find_vocab_size(self, tokenizer):
        return tokenizer.get_vocab_size()

    def __call__(self, batch_size, **kwargs):

        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "[START] " + dataset[i, 0] + " [END]"
            dataset[i, 1] = "[START] " + dataset[i, 1] + " [END]"
		
        dataset = Dataset.from_tensor_slices([self.encoder_voc, self.decoder_voc]).shuffle(self.n_sentences)
        train = dataset.take(self.train_split*self.n_sentences).batch(64)
        test = dataset.skip(self.train_split*self.n_sentences).batch(64)	

        enc_tokenizer = self.create_tokenizer(self.encoder_voc)
        enc_seq_length = self.find_seq_length(self.encoder_voc)

        dec_tokenizer = self.create_tokenizer(self.decoder_voc)
        dec_seq_length = self.find_seq_length(self.decoder_voc)

        return train, test, enc_tokenizer, dec_tokenizer, enc_seq_length, dec_seq_length, self.n_sentences

import nltk

class Noun_extractor:
    #noun extractor
    def __init__(self):
        self.sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
                
            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
            """
        self.chunker = nltk.RegexpParser(self.grammar)

    def get_noun(self,text=''):
        # extract the noun from a sentence and return a set
        from nltk.corpus import stopwords
        global stopwords
        nouns = set()
        if text =='':
            print "Empty sentence!"
            return 0
        else:
            toks = nltk.regexp_tokenize(text, self.sentence_re)
            postoks = nltk.tag.pos_tag(toks)
            print "postoks=:"
            print postoks
            tree = self.chunker.parse(postoks)
            stopwords = stopwords.words('english')
            terms = self.get_terms(tree)
            for term in terms:
                for word in term:
                    nouns.add(str(word))
            return nouns
                    
    
    def leaves(self,tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def normalise(self,word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = self.lemmatizer.lemmatize(word)
        return word

    def acceptable_word(self, word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(2 <= len(word) <= 40
            and word.lower() not in stopwords)
        return accepted


    def get_terms(self, tree):
        for leaf in self.leaves(tree):
            term = [ self.normalise(w) for w,t in leaf if self.acceptable_word(w) ]
            yield term


if __name__ == '__main__':
    text = raw_input("Enter the text please ...")
    extractor = Noun_extractor()
    nouns = extractor.get_noun(text)
    print "The nouns are:"
    print nouns 
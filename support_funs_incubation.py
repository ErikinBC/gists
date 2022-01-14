import sys
import re
import numpy as np
import pandas as pd
import itertools

from spacy.lang.en import English
en = English()
en.add_pipe(en.create_pipe('sentencizer'))  # updated

def stopifnot(cond, stmt=None):
    if stmt is None:
        stmt = 'error! condition net met'
    else:
        try:
            stopifnot(isinstance(stmt, str))
        except:
            sys.exit('stmt is not a string')
    if not cond:
        sys.exit(stmt)


# Joins list of lists into one list
def ljoin(lst):
    lst = [z for z in lst if z is not None]
    lst = list(itertools.chain.from_iterable(lst))
    return lst


# approximate pd.Series.str.contains()
def idx_find(lst, pat):
    idx = np.repeat(False, len(lst))
    for ii, aa in enumerate(lst):
        if isinstance(aa, float):  # nan values
            continue
        idx[ii] = re.search(pat, aa, re.IGNORECASE) is not None
    return idx


# Find word preceeding or succeeding the pattern
def find_beside(sentence, pat, tt='right'):
    stopifnot(tt in ['right', 'left'])
    if tt == 'right':
        rpat = pat + '\s\w+'
    else:
        rpat = '\w+\s' + pat
    if re.search(rpat, sentence, re.IGNORECASE) is None:
        return None
    fi = re.finditer(rpat, sentence, re.IGNORECASE)
    holder = []
    spat = r'\s?' + pat + r'\s?'
    for z in fi:
        val = re.sub(spat, '', z.group(), flags=re.IGNORECASE)
        holder.append(val)
    return holder


# Returns sentence with key-word
def word_find(para, pat):
    lpara = para.lower()
    pat = pat.lower()
    holder = []
    if re.search(pat, lpara):
        z = re.finditer(pat, lpara)
        holder.append([x.group() for x in z])
    holder = ljoin(holder)
    return holder


# Find number of matching words
def uwords(corpus, pat):
    holder = []
    for para in corpus:
        val = word_find(para, pat)
        if len(val) > 0:
            holder.append(val)
    holder = ljoin(holder)
    dat = pd.Series(holder).value_counts().reset_index()
    dat.rename(columns={0: 'n', 'index': 'term'}, inplace=True)
    return dat


# Function to find sentence with a pattern
def sentence_find(corpus, pat):
    if isinstance(pat, str):
        pat = [pat]  # Force to list otherwise
    sentences = en(corpus)
    holder = []
    for sentence in sentences.sents:
        ss = sentence.text
        for pp in pat:
            search = re.search(pp, ss, re.IGNORECASE)
            if search is not None:
                fi = re.finditer(pp, ss, re.IGNORECASE)
                idx = []
                for z in fi:
                    idx.append(z.span())
                holder.append((idx, ss))
    return holder

# Loops through a sentence_find() output and allows user to pick useful sentences
def record_vals(idx_sentences):
    holder = []
    jj, cond = 0, True
    n_sentences = len(idx_sentences)
    for jj in range(n_sentences):
        ii_ss = idx_sentences[jj]
        print('Sentence %i of %i\n' % (jj+1, n_sentences))
        color_printer(ii_ss)
        choice = input().lower()
        if choice in ['y', 'yes']:
            holder.append(ii_ss)
        if choice in ['esc', 'break', 'exit']:
            break
        if jj+1 == len(idx_sentences):
            break
    return holder

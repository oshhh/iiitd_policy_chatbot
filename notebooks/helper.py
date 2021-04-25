import json
import csv
from docx import Document
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy
import neuralcoref
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer 

nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)
stemmer = PorterStemmer() 


def read_text(filename):
    raw_text = ''
    with open(filename) as file:
        for line in file:
            raw_text += line
    return raw_text

def write_text(text, filename):
    with open(filename, 'w') as file:
        for line in text:
            file.write(line)
            
def read_json(filename):
    with open(filename) as file:
        data = json.load(file)
    return data

def write_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

def read_csv(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.append(row)
    return data

def write_csv(data, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


sentence_1 = '(:?(?:(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*)?)*),)?(?:N+(?:[C,]+N+)*)(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?:[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?:N+(?:[C,]+N+)*)?,?(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))'
sentence_with_groups = '(?P<sentence>(?:(?P<modifiers0>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*)?)*),)?(?P<subject>N+(?:[C,]+N+)*)(?P<modifiers1>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?P<relation>[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?P<object>N+(?:[C,]+N+)*)?,?(?P<modifiers2>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))'
sentences_2_wo_groups = '(?:(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*)?)*),)?(?:N+(?:[C,]+N+)*)(?:>(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?:[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?:N+(?:[C,]+N+)*)?(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))(?:(?:[C,](?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?:[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?:N+(?:[C,]+N+)*)?(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))*)'


sentences_1 = '(' + sentence_1 + ')(:[C,]+' + sentence_1 + ')*'
sentences_2 = '(?:(?P<modifiers0>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*)?)*),)?(?P<subject>N+(?:[C,]+N+)*)(?P<sentence1>(?P<modifiers1>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?P<relation>[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?P<object>N+(?:[C,]+N+)*)?(?P<modifiers2>((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))(?P<remaining>(?:[C,](?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*)(?:[VA]*V+[VA]*(?:[C,]+[VA]*V+[VA]*)*)(?:N+(?:[C,]+N+)*)?(?:((?:A+(?:[C,]+A+)*)(?:N+(?:[C,]+N+)*))*))*)'

conditional_sentences_1 = 'I(?P<condition>' + sentences_1 + '|' + sentences_2_wo_groups + '),?(?P<statement>' + sentences_1 + '|' + sentences_2_wo_groups + ')'
conditional_sentences_2 = '(?P<statement>' + sentences_1 + '|' + sentences_2_wo_groups + '),?I(?P<condition>' + sentences_1 + '|' + sentences_2_wo_groups + ')'

regex_compiled = False
re_sentences = None
re_sentences2 = None
re_conditional_sentences_1 = None
re_conditional_sentences_2 = None
re_sentence = None
re_modifiers = None
re_compound = None
cse = None
csam = None
csd = None
csss = None
csb = None
csai = None
ece = None
dosa = None
doaa = None
example = None
ie = None
etc = None
number = None
btech = None

def compile_regex():
    global regex_compiled
    global re_sentences
    global re_sentences2
    global re_conditional_sentences_1
    global re_conditional_sentences_2
    global re_sentence
    global re_modifiers
    global re_compound
    global cse
    global csam
    global csd
    global csss
    global csb
    global csai
    global ece
    global dosa
    global doaa
    global example
    global ie
    global etc
    global number
    global btech
    
    if regex_compiled:
        return
    regex_compiled = True
    re_sentences = re.compile(sentences_1)
    re_sentences2 = re.compile(sentences_2)
    re_conditional_sentences_1 = re.compile(conditional_sentences_1)
    re_conditional_sentences_2 = re.compile(conditional_sentences_2)

    re_sentence = re.compile(sentence_with_groups)
    re_modifiers = re.compile('(?P<modifier>(?P<m_rel>A+(?:[C,]+A+)*)(?P<m_obj>N+(?:[C,]+N+)*))(?P<remaining>[C,]?(?:A+(?:[C,]+A+)*N+(?:[C,]+N+)*)*)')
    re_compound = re.compile('(?P<first>[NVA]+)(?P<remaining>(?:[C,]+[NVA]+)*)')

    cse = re.compile('computer science and engineering', re.IGNORECASE)
    csam = re.compile('computer science and applied mathematics', re.IGNORECASE)
    csd = re.compile('computer science and design', re.IGNORECASE)
    csss = re.compile('computer science and social sciences', re.IGNORECASE)
    csb = re.compile('computer science and biosciences', re.IGNORECASE)
    csai = re.compile('computer science and artificial intelligence', re.IGNORECASE)
    ece = re.compile('electronics and communications engineering', re.IGNORECASE)
    dosa = re.compile('dean of student affairs', re.IGNORECASE)
    doaa = re.compile('dean of academic affairs', re.IGNORECASE)

    example = re.compile('e\.g\.', re.IGNORECASE)
    ie = re.compile('i\.e\.', re.IGNORECASE)
    etc = re.compile('etc\.', re.IGNORECASE)
    number = re.compile('no\.', re.IGNORECASE)
    btech = re.compile('b[\.]?tech[\.]?', re.IGNORECASE)

def preprocess_paragraph(text):
    compile_regex()
    temp = ''
    for token in text.split():
        if token[:4] == 'http':
            temp += token + ' '
        else:
            temp += token.replace('/', ' or ') + ' '
    preprocessed_text = ''
    for char in temp:
        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,!?.[]{}\'’\"@#%^&*()-_=+\\;:<>/1234567890 ':
            preprocessed_text += char
    para = nlp(preprocessed_text)._.coref_resolved
    para = cse.sub('cse', para)
    para = csam.sub('csam', para)
    para = csd.sub('csd', para)
    para = csss.sub('csss', para)
    para = csb.sub('csb', para)
    para = csai.sub('csai', para)
    para = ece.sub('ece', para)
    para = doaa.sub('doaa', para)
    para = dosa.sub('dosa', para)
    
    para = example.sub('example', para)
    para = ie.sub('ie', para)
    para = etc.sub('etc', para)
    para = number.sub('number', para)
    para = btech.sub('btech', para)
    return para

def split_into_sentences(text):
#     dont_split = set(['mr', 'ms', 'mrs', 'etc', 'dr', 'no', 'e.g'])
#     sentences = []
#     sentence = []
#     for c in range(len(text)):
#         if text[c] == '.' and c + 2 < len(text) and text[c + 2].lower() != text[c + 2] and nlp(''.join(sentence))[-1].text.lower() not in dont_split:
#                 sentences.append(''.join(sentence))
#                 sentence = []
#         else:
#             sentence.append(text[c])
#     sentences.append(''.join(sentence))
#     return sentences
    return sent_tokenize(text)

def get_sentence_structure(sentence):
    pos_ind = {
        'NOUN': 'N', 
        'PROPN': 'N', 
        'ADJ': 'N', 
        'DET': 'N', 
        'PART': 'N', 
        'NUM': 'N', 
        'PRON': 'N',
        'AUX': 'V', 
        'VERB': 'V', 
        'ADV': 'A',
        'ADP': 'A',
        'CCONJ': 'C', 
        'SCONJ': 'C',
        'COMMA': ',',
        'PUNCT': '',
        'SYM': '',
        'SPACE': '',
        'X': '',
        'INTJ': '',
        'if': 'I'
    }
    sent = nlp(sentence)
    
    pos_tags = []
    tokens = []
    verb_is_A = False
    for i in range(len(sent)):
        if pos_tags and pos_tags[-1] != pos_ind['ADV'] and verb_is_A:
                verb_is_A = False
        if sent[i].text.lower() == 'if':
            pos_tags.append(pos_ind['if'])
            tokens.append(sent[i].text)
            continue
        if sent[i].text.lower() in ['which', 'who']:
            pos_tags.append(pos_ind['ADV'])
            tokens.append(sent[i].text)
            verb_is_A = True
            continue
        if ',' in sent[i].text and sent[i].pos_ == 'PUNCT':
            pos_tags.append(pos_ind['COMMA'])
            tokens.append(sent[i].text)
            continue

        if sent[i].text == 'not' or sent[i].text == 'to':
            pos_tags.append(pos_ind['ADV'])
            tokens.append(sent[i].text)
            continue
        
        if pos_ind[sent[i].pos_] == 'V' and verb_is_A:
            pos_tags.append(pos_ind['ADP'])
            tokens.append(sent[i].text)
            continue
        if sent[i].pos_ == 'VERB' and sent[i].tag_ == 'VBN' and (i == 0 or sent[i - 1].pos_ != 'AUX') and i + 1 < len(sent) and sent[i + 1].pos_ == 'ADP':
            pos_tags.append(pos_ind['ADP'])
            tokens.append(sent[i].text)
            continue

        if sent[i].pos_ == 'VERB' and sent[i].tag_ == 'VBG' and (i == 0 or sent[i - 1].pos_ != 'AUX') and i + 1 < len(sent) and sent[i + 1].pos_ == 'ADP':
            pos_tags.append(pos_ind['ADP'])
            tokens.append(sent[i].text)
            continue
        if sent[i].pos_ == 'VERB' and sent[i].tag_ == 'VBG' and (i == 0 or sent[i - 1].pos_ != 'AUX') and i + 1 < len(sent) and sent[i + 1].pos_ != 'ADP':
            pos_tags.append(pos_ind['ADJ'])
            tokens.append(sent[i].text)
            continue
        if sent[i].text.lower() == 'because' and i < len(sent) and sent[i + 1].text.lower() == 'of':
            pos_tags.append('A')
            tokens.append(sent[i].text)
            continue
#         print(sent[i].text, sent[i].pos_)
        if pos_ind[sent[i].pos_]:
            pos_tags.append(pos_ind[sent[i].pos_])
            tokens.append(sent[i].text)
    
    tokens_ = []
    sentence_structure = []
    i = 0
    while i < len(pos_tags):
        if not pos_tags[i] in [pos_ind['COMMA'], pos_ind['CCONJ'], pos_ind['SCONJ']] or i + 1 < len(pos_tags) and not pos_tags[i + 1] in [pos_ind['COMMA'], pos_ind['CCONJ'], pos_ind['SCONJ']]:
            sentence_structure.append(pos_tags[i])
            tokens_.append(tokens[i])
        i += 1
    sentence_structure = ''.join(sentence_structure)
    return sentence_structure, tokens_

    
def extract(sentence):
    compile_regex()
    # remove text in brackets
    preprocessed_sent = ''
    in_bracket = 0
    for char in sentence:
        if char == ')':
            in_bracket -= 1
            continue
        if char == '(':
            in_bracket += 1
            continue
        if in_bracket:
            continue
        preprocessed_sent += char
    sentence = preprocessed_sent
    
    # words that hinder extraction
    problem_words = ['such as', 'in which', 'where', 'when']
    for problem_word in problem_words:
        if problem_word in sentence.lower():
            return []
    # get the structure of the sentence using pos tags
    sentence_structure, tokens = get_sentence_structure(sentence)
    # extract from structure
    extractions = find_match(sentence_structure, tokens)
    # change the format of output extractions to ensure a single subject, relation and object in each sentence
    final_extractions = []
    for extraction in extractions:
        for sub in extraction['subject']:
            for rel in extraction['relation']:
                for obj in extraction['object']:
                    ext = {
                        'subject': sub,
                        'relation': rel,
                        'object': obj,
                        'modifiers': [{'m_rel': m_rel, 'm_obj': m_obj} for mod in extraction['modifiers0'] + extraction['modifiers2'] for m_rel in mod['m_rel'] for m_obj in mod['m_obj']],
                        'subject_modifiers': [{'m_rel': m_rel, 'm_obj': m_obj} for mod in extraction['modifiers1'] for m_rel in mod['m_rel'] for m_obj in mod['m_obj']],
                        'condition': extraction['condition']
                    }
                    final_extractions.append(ext)
    return final_extractions


def find_match(sentence_structure, text):
    extractions = []
    if re_conditional_sentences_1.fullmatch(sentence_structure):
        match = re_conditional_sentences_1.fullmatch(sentence_structure)
        extractions = find_match(sentence_structure[match.start('statement'): match.end('statement')], text[match.start('statement'): match.end('statement')])
        for ext in extractions:
            ext['condition'] = ' '.join(text[match.start('condition'): match.end('condition')])
    elif re_conditional_sentences_2.fullmatch(sentence_structure):
        match = re_conditional_sentences_2.fullmatch(sentence_structure)
        extractions = find_match(sentence_structure[match.start('statement'): match.end('statement')], text[match.start('statement'): match.end('statement')])
        for ext in extractions:
            ext['condition'] = ' '.join(text[match.start('condition'): match.end('condition')])
    elif re_sentences.fullmatch(sentence_structure):
        while text:
            match = re_sentences.fullmatch(sentence_structure)
            extractions.append(break_sentence(sentence_structure[match.start(1): match.end(1)], text[match.start(1): match.end(1)]))
            text = text[match.end(1) + 1:]
            sentence_structure = sentence_structure[match.end(1) + 1:]
    elif re_sentences2.fullmatch(sentence_structure):
        while text:
            match = re_sentences2.fullmatch(sentence_structure)
            if not match:
                break
            extractions.append(break_sentence(sentence_structure[:match.start('remaining')], text[:match.start('remaining')]))
            sentence_structure = sentence_structure[:match.start('sentence1')] + sentence_structure[match.end('sentence1') + 1:]
            text = text[:match.start('sentence1')] + text[match.end('sentence1') + 1:]
    return extractions
        
def break_sentence(sentence_structure, text):
    match = re_sentence.fullmatch(sentence_structure)
    extraction = {}
    extraction['subject'] = break_compound(sentence_structure[match.start('subject'): match.end('subject')], text[match.start('subject'): match.end('subject')])
    extraction['relation'] = break_compound(sentence_structure[match.start('relation'): match.end('relation')], text[match.start('relation'): match.end('relation')])
    extraction['object'] = break_compound(sentence_structure[match.start('object'): match.end('object')], text[match.start('object'): match.end('object')])
    extraction['modifiers0'] = break_modifiers(sentence_structure[match.start('modifiers0'): match.end('modifiers0')], text[match.start('modifiers0'): match.end('modifiers0')])
    extraction['modifiers1'] = break_modifiers(sentence_structure[match.start('modifiers1'): match.end('modifiers1')], text[match.start('modifiers1'): match.end('modifiers1')])
    extraction['modifiers2'] = break_modifiers(sentence_structure[match.start('modifiers2'): match.end('modifiers2')], text[match.start('modifiers2'): match.end('modifiers2')])
    extraction['condition'] = None
    return extraction

def break_modifiers(modifiers, tokens):
    if not modifiers:
        return []
    match = re_modifiers.fullmatch(modifiers)
    if not match:
        return []
    modifier = {
        'm_rel': break_compound(modifiers[match.start('m_rel'): match.end('m_rel')], tokens[match.start('m_rel'): match.end('m_rel')]),
        'm_obj':break_compound(modifiers[match.start('m_obj'): match.end('m_obj')], tokens[match.start('m_obj'): match.end('m_obj')])
    }
    modifiers = break_modifiers(modifiers[match.start('remaining'): match.end('remaining')], tokens[match.start('remaining'): match.end('remaining')])
    return [modifier] + modifiers

def break_compound(compound, tokens):
    if not compound:
        return []
    match = re_compound.fullmatch(compound)
    if not match:
        return []
    parts = break_compound(compound[match.start('remaining') + 1:], tokens[match.start('remaining') + 1:])
    return [' '.join(tokens[match.start('first'): match.end('first')])] + parts

def get_type(entity):
    words = list(entity.split())
    
    # links
    if 'http' in entity:
        return entity, ['link', 'web']
    if '@' in entity and ('.com' in entity or '.in'):
        return entity, ['link', 'mail']
    if 'erp' in words:
        return 'erp', ['link']
    
    # committee check
    if 'student' in words and 'council' in words:
        return 'student council', ['committee']
    if 'senate' in words:
        return 'student senate', ['committee']
    if 'disciplin' in entity and 'committee' in words or 'dac' in words:
        return 'disciplinary action committee', ['committee']
    if 'ugc' in words or 'ug' in words and 'committee' in words or 'under' in words and 'committee' in words:
        return 'undergraduate committee', ['committee']
    if 'committee' in words or 'council' in words:
        return entity, ['committee']
    # location check
    if 'gate' in words:
        if '1' in entity:
            return 'gate 1', ['gate']
        if '2' in entity:
            return 'gate 2', ['gate']
        if '3' in entity:
            return 'gate 3', ['gate']
        return entity, ['gate']
    if 'hostel' in words:
        return 'hostel', ['location', 'building']
    if 'library' in words:
        return 'library', ['location', 'building']
    if 'sport' in entity and 'complex' in words:
        return 'sports block', ['location', 'building']
    if 'canteen' in words or 'mess' in words or 'dining' in words:
        return 'canteen', ['location', 'building', 'canteen']
    if 'block' in words or 'building' in words:
        if 'sem' in entity:
            return 'seminar block', ['location', 'building']
        if 'new' in entity and 'acad' in entity or 'r&d' in entity or 'research' in entity:
            return 'new academic block', ['location', 'building']
        if 'acad' in entity and 'block' in entity:
            return 'old academic block', ['location', 'building']
        if 'sport' in entity:
            return 'sports block', ['location', 'building']
        return entity, ['location', 'building']
    if 'room' in words or 'floor' in words or 'lab' in words or 'hall' in words:
        return entity, ['location']

    # person checks
    if 'doaa' in words or 'dosa' in words or 'dean' in words or 'chair' in entity:
        return entity, ['person', 'faculty', entity]
    if 'prof' in entity or 'faculty' in words or 'instructor' in words:
        return 'faculty', ['person', 'faculty']
    if 'staff' in words:
        return 'staff', ['person', 'staff']

    if 'student' in words:
        if 'btech' in entity or 'b.tech' in entity or 'bachelor' in entity or 'ug' in entity:
            return 'btech student', ['person', 'student']
        if 'mtech' in entity or 'm.tech' in entity or 'master' in entity or 'pg' in entity:
            return 'mtech student', ['person', 'student']
        if 'phd' in entity:
            return 'phd student', ['person', 'student']
        if 'hostel' in entity:
            return 'hostel student', ['person', 'student']
        if words[-1] == 'student':
            return entity, ['person', 'student']
    if 'hosteller' in words:
        return 'hostel student', ['person', 'student']
    if 'parent' in words or 'guardian' in words:
        return 'parent', ['person', 'parent']
    if 'visitor' in words:
        return 'visitor', ['person', 'visitor']

    # credit
    if 'credit' in entity:
        return entity, ['number', 'credit']
    # date
    months = ['jan', 'january', 'feb', 'february', 'mar', 'march', 'apr', 'april', 'may', 'jun', 'june', 'jul', 'july', 'aug', 'august', 'sep', 'sept', 'september', 'oct', 'october', 'nov', 'november', 'dec', 'december']
    for month in months:
        if month in words:
            return entity, ['datetime']
    if re.match('.*20[0-9][0-9]-.*', entity) or re.match('.*19[0-9][0-9]-.*', entity) or re.match('.*-20[0-9][0-9].*', entity) or re.match('.*/20[0-9][0-9].*', entity):
        return entity, ['datetime']
    # time
    if 'year' in words or 'month' in words or 'week' in words or 'day' in words or 'hr' in words or 'hrs' in words or 'hour' in words or 'mins' in words or 'minute' in words or 'sec' in words or 'second' in words or 'am' in words or 'pm' in words:
        return entity, ['datetime']
    # money
    if 'rs' in words:
        return entity, ['number', 'money']
    # number
    isNumber = False
    for num in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
        if num in words:
            return entity, ['number']
    for char in entity:
        if char in '1234567890':
            return entity, ['number']
    if 'number' in words:
        return entity, ['number']

    # department check
    if 'design' in words or 'csd' in words:
        return 'csd', ['department']
    if 'electronics' in words or 'ece' in words:
        return 'ece', ['department']
    if 'biology' in words or 'csb' in words:
        return 'cdb', ['department']
    if 'math' in entity or 'csam' in words:
        return 'csam', ['department']
    if 'social science' in entity or 'csss' in words:
        return 'csss', ['department']
    if 'computer science' in entity or 'cse' in words:
        return 'cse', ['department']

    # program check
    if 'b.tech' in entity or 'btech' in entity or 'bachelor' in entity or 'ug' in words or 'undergrad' in entity:
        return 'btech program', ['program']
    if 'm.tech' in entity or 'mtech' in entity or 'master' in entity or 'pg' in words:
        return 'mtech program', ['program']
    if 'phd' in words:
        return 'phd program', ['program']
    if 'program' in words:
        return entity, ['program']

    # course check
    if 'sg' in words or 'self growth' in entity:
        return 'self growth', ['course']
    if 'cw' in words or 'community work' in entity:
        return 'community work', ['course']
    if 'course' in entity or 'btp' in entity or 'b.tech project' in entity or 'ip' in words or 'independent project' in entity:
        return entity, ['course']

    # gpa
    if entity == 'cgpa':
        return 'cgpa', ['grades', 'gpa']
    if entity == 'sgpa':
        return 'sgpa', ['grades', 'gpa']
    if 'grade' in entity or 'gpa' in entity:
        return entity, ['grades']
    
    # registration
    if 'registration' in words:
        return entity, ['registration']
    # admissions
    if 'admission' in words:
        return entity, ['admission']
    # graduation
    if 'graduat' in entity:
        return entity, ['graduation']
    # evaluation
    if 'test' in words:
        return 'test', ['evaluation']
    if 'midsem' in words or ('mid' in entity and 'exam' in entity):
        return 'midsem', ['evaluation']
    if 'endsem' in words or ('end' in entity and 'exam' in entity):
        return 'endsem', ['evaluation']
    if 'quiz' in words:
        return 'quiz', ['evaluation']
    if 'assignment' in words:
        return 'assignment', ['evaluation']
    if 'evaluation' in words or 'exam' in words:
        return entity, ['evaluation']
    # vacation
    if 'vacation' in words or 'recess' in words or 'holiday' in words:
        return entity, ['holiday']
    # semester
    if 'semester' in words or 'term' in words:
        if 'monsoon' in words:
            return 'monsoon semester', ['semester']
        elif 'winter' in words:
            return 'winter semester', ['semester']
        elif 'summer' in words:
            return 'summer semester', ['semester']
        return entity, ['semester']
    # fees and charges
    if 'fee' in words:
        return entity, ['fee']
    # org check
    if 'iiit' in entity or 'campus' in words or 'institute' in words or 'college' in words:
        return 'iiitd', ['org']
    
    return entity, []

def entity_canonicalisation(entity):
    entity = entity.lower()
    tags = []
    tokens = []
    for token in nlp(entity):
        if token.pos_ == 'DET' or token.pos_ == 'PRON': 
            continue
        if token.tag_ in {"NNS", "NNPS"}:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)
    if tokens:
        entity, tags = get_type(' '.join(tokens))
    return entity, tags, tokens


def canonicalise(extractions):
    for ext in extractions:
        ext = extractions[ext]
        
        # relation synsets
        rel_synsets = set([])
        rel_root = set([])
        if ext['object']:
            sentence = ext['subject'] + ' ' + ext['relation'] + ext['object']
        else:
            sentence = ext['subject'] + ' ' + ext['relation']
        doc = nlp(ext['relation'])
        for token in doc:
            if token.pos_ == 'VERB' and token.text not in ['will', 'shall', 'may', 'must', 'can', 'could']:
                try:
                    rel_synsets.add(lesk(sentence, token.text, 'v').name())
                    rel_root.add(token.lemma_)
                except:
                    print('ERROR:', token.lemma_,)
        ext['rel_synsets'] = list(rel_synsets)
        
        # entity canonicalisation
        ext['subject'] = entity_canonicalisation(ext['subject'])
        ext['object'] = entity_canonicalisation(ext['object'])
        for m in ext['modifiers'] + ext['subject_modifiers']:
            m['m_obj'] = entity_canonicalisation(m['m_obj'])
    
    return extractions

def find_keywords(query):
    sentence_structure, tokens = get_sentence_structure(query)
    keywords = []
    i = 0
    keyword = []
    while i < len(tokens):
        while i < len(tokens):
            if sentence_structure[i] != 'N':
                i += 1
            else:
                break
        while i < len(tokens):
            if sentence_structure[i] == 'N':
                keyword.append(tokens[i])
                i += 1
            else:
                break
        if keyword:
            keyword, tags, words = entity_canonicalisation(' '.join(keyword))
            keywords.append((keyword, tags, words))
        keyword = []
    return keywords

def get_stemmed_sentence_tokens(sentence):
    words = [word.text for word in nlp(sentence) if word.pos_ in ['NOUN', 'VERB', 'ADV']]
    return list(set([stemmer.stem(word) for word in words]))

question_types = {
    'when': 'datetime',
    'where': 'location',
    'who': 'person',
    'how many': 'number',
}

def get_question_type(query):
    global question_types
    sentences = split_into_sentences(query)
    q_types = []
    for question in sentences:
        for q_type in question_types:
            if question.lower()[:len(q_type)] == q_type:
                q_types.append(question_types[q_type])
    return q_types


def plot_graph(graph, filename):
    nx_graph = nx.DiGraph()
    nodes = {}
    type_count = 0
    node_color = [0]
    nx_graph.add_node('True')
    for type in graph['vertices']:
        type_count += 1
        for node in graph['vertices'][type]:
            nx_graph.add_node(node)
            node_color.append(type_count)
    for edge in graph['edges']:
        nx_graph.add_edge(edge[0], edge[2], **{'label': edge[1]})

    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, node_color = node_color, with_labels=True, font_size = 10)
    edge_labels = nx.get_edge_attributes(nx_graph,'label')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels = edge_labels, font_size = 10)
    
    figure = plt.gcf()
    figure.set_size_inches(50, 50)
    plt.savefig(filename, dpi=200)



import time
import numpy as np

from helper import *

from neo4j import GraphDatabase
driver = None

mrc = None


bert_model = None
bert_tokenizer = None


def init_kg():
    global driver
    driver = GraphDatabase.driver('neo4j://localhost:11003', auth=('chatbot', 'password'))

def init_bert():
    global bert_model
    global bert_tokenizer
    from transformers import BertForQuestionAnswering
    from transformers import BertTokenizer
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def init_mrc():
    global mrc
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.rc
    mrc = Predictor.from_path("../bidaf-elmo-model-2020.03.19.tar.gz")

def shortlist_sentences(query):
    global driver

    start = time.time()

    with driver.session() as session:
        sentences = session.read_transaction(q_shortlist_sentences, query, 20)

    return sentences

def find_answer_from_mrc(question, sentences):
    global mrc
    if not mrc:
        raise 'mrc not initialised'

    passage = '\n'.join(sentence['topic'] + ': ' + sentence['sentence'] for sentence in sentences)
    answers = mrc.predict(question, passage)["best_span_str"].split('.')
    final_answer = []
    for answer in answers:
        for sentence in sentences:
            if answer in sentence['topic'] + ': ' + sentence['sentence']:
                final_answer.append(sentence)

    return final_answer

def find_answer_from_bert(question, sentences):
    global bert_model
    global bert_tokenizer
    if not bert_model:
        raise 'bert not initialised'

    paragraph = '\n'.join(sentence['topic'] + ': ' + sentence['sentence'] for sentence in sentences)

    encoding = bert_tokenizer.encode_plus(text=question,text_pair=paragraph, add_special=True)

    inputs = encoding['input_ids']
    sentence_embedding = encoding['token_type_ids']
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs)

    scores = bert_model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    
    start_index = torch.argmax(scores.start_logits)
    end_index = torch.argmax(scores.end_logits)
    answers = (' '.join(tokens[start_index:end_index+1])).split()

    final_answer = []
    for answer in answers:
        for sentence in sentences:
            if answer in sentence['topic'] + ': ' + sentence['sentence']:
                final_answer.append(sentence)

    return final_answer


def q_shortlist_sentences(tx, query, top_k):
    global question_types
    
    keywords = [list(keyword) for keyword in find_keywords(query)]
    print(keywords)
    
    question_types = get_question_type(query)
    print(question_types)
    
    stemmed_tokens = get_stemmed_sentence_tokens(query)
    
    keywords.append(['##NO_MATCH##', question_types, []])
    
    weights = [ 0.3, 1.01868511, 0,  0.28790363,  0.38202574, 0, 0.28790363,  0.37467432,  0.56424285,  0.        ]
    query = (
        'with ' + str(keywords) + ' as keywords, ' + str(question_types) + ' as answer_types, ' + str(stemmed_tokens) + ' as stemmed_query_tokens \n' +
        'match (main_topic:Topic)<-[]-(p:Paragraph)-[]->(s:Sentence)-[*]->(sent_e:ExtEntity) \n' + 
        'match path = (t:Topic)<-[:about_topic*1..2]-(p) \n' +
        'with keywords, answer_types, stemmed_query_tokens, main_topic, p, s, collect(distinct sent_e) as entities, collect(distinct [t, length(path)]) as topics \n' +
        'match (main_topic)<-[*]-(:Paragraph)-->(nbr_s:Sentence) \n' +
        'where abs(nbr_s.id - s.id) <= 2 \n' +
        'match (nbr_s)-[*]->(nbr_e:ExtEntity) \n' + 
        'with keywords, answer_types, stemmed_query_tokens, p, s, main_topic as topic, topics, entities, collect(distinct nbr_e) as nbr_entities \n' +
        'with keywords, answer_types, p, s, topic.text as topic, topics, entities, nbr_entities, s.id as s_id, s.text as sentence, size(apoc.coll.intersection(s.stemmed_tokens, stemmed_query_tokens)) as sent_stemmed_overlap, reduce( \n' + 
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'distance = 0.0, entity in entities | \n' + 
                'case \n' + 
                    'when apoc.text.levenshteinDistance(entity.text, keyword[0])/toFloat(size(entity.text) + size(keyword[0])) <= 0.1 and distance < entity.idf \n' + 
                    'then entity.idf \n' + 
                    'else distance \n' + 
                'end \n' + 
            ') \n' + 
        ') as sent_text, reduce( \n' + 
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'jaccard = 0.0, entity in entities | \n' + 
                'case \n' +
                    'when entity.idf * toFloat(size(apoc.coll.intersection(entity.tags, keyword[1]))) > jaccard \n' + 
                    'then entity.idf * toFloat(size(apoc.coll.intersection(entity.tags, keyword[1]))) \n' +
                    'else jaccard \n' +
                'end \n' +
            ') \n' + 
        ') as sent_tags, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'jaccard = 0.0, entity in entities | \n' + 
                'case \n' +
                    'when entity.idf * toFloat(size(apoc.coll.intersection(entity.tokens, keyword[2]))) > jaccard \n ' + 
                    'then entity.idf * toFloat(size(apoc.coll.intersection(entity.tokens, keyword[2]))) \n' +
                    'else jaccard \n' +
                'end \n' +
            ') \n' + 
        ') as sent_tokens, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'distance = 0.0, entity in nbr_entities | \n' + 
                'case \n' + 
                    'when apoc.text.levenshteinDistance(entity.text, keyword[0])/toFloat(size(entity.text) + size(keyword[0])) <= 0.1 and distance < entity.idf \n' + 
                    'then entity.idf \n' + 
                    'else distance \n' + 
                'end \n' + 
            ') \n' + 
        ') as nbr_text, reduce( \n' + 
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'jaccard = 0.0, entity in entities | \n' + 
                'case \n' +
                    'when entity.idf * toFloat(size(apoc.coll.intersection(entity.tags, keyword[1]))) > jaccard \n' + 
                    'then entity.idf * toFloat(size(apoc.coll.intersection(entity.tags, keyword[1]))) \n' +
                    'else jaccard \n' +
                'end \n' +
            ') \n' + 
        ') as nbr_tags, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'jaccard = 0.0, entity in entities | \n' + 
                'case \n' +
                    'when entity.idf * toFloat(size(apoc.coll.intersection(entity.tokens, keyword[2]))) > jaccard \n' + 
                    'then entity.idf * toFloat(size(apoc.coll.intersection(entity.tokens, keyword[2]))) \n' +
                    'else jaccard \n' +
                'end \n' +
            ') \n' + 
        ') as nbr_tokens, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'topics_matched = 0.0, topic in topics | \n' +
                'case \n' +
                'when topic[1] = 1 then \n'
                    'topics_matched + size(apoc.coll.intersection(topic[0].tags, keyword[1])) + reduce( \n' +
                        'distance = 0.0, entity in topic[0].keywords | \n' + 
                        'case \n' + 
                            'when apoc.text.levenshteinDistance(entity, keyword[0])/toFloat(size(entity) + size(keyword[0])) <= 0.1 \n' + 
                            'then distance + 1 - apoc.text.levenshteinDistance(entity, keyword[0])/toFloat(size(entity) + size(keyword[0])) \n' + 
                            'else distance \n' + 
                        'end \n' + 
                    ') \n' +
                'else topics_matched \n' +
                'end \n'
            ') \n' + 
        ') as topic1, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'topics_matched = 0.0, topic in topics | \n' +
                'case \n' +
                'when topic[1] = 2 then \n'
                    'topics_matched + size(apoc.coll.intersection(topic[0].tags, keyword[1])) + reduce( \n' +
                        'distance = 0.0, entity in topic[0].keywords | \n' + 
                        'case \n' + 
                            'when apoc.text.levenshteinDistance(entity, keyword[0])/toFloat(size(entity) + size(keyword[0])) <= 0.1 \n' + 
                            'then distance + 1 - apoc.text.levenshteinDistance(entity, keyword[0])/toFloat(size(entity) + size(keyword[0])) \n' + 
                            'else distance \n' + 
                        'end \n' + 
                    ') \n' +
                'else topics_matched \n' +
                'end \n'
            ') \n' + 
        ') as topic2, reduce( \n' +
            'contains_answer_type = 0.0, type in answer_types | \n' +
            'case \n' +
                'when contains_answer_type = 0 and reduce( \n' +
                        'contains_type = false, entity in entities | \n' +
                        'case \n' +
                            'when contains_type = 0 then (entity.tags contains type) \n' +
                            'else contains_type \n' +
                        'end \n' +
                    ') \n'
                'then 1.0 \n'
                'else contains_answer_type \n' +
            'end \n' +
        ') as answer_type \n' +
        'return s_id, sentence, topic, topics, sent_stemmed_overlap, sent_text, sent_tags, sent_tokens, nbr_text, nbr_tags, nbr_tokens, topic1, topic2, answer_type'
    )

    result = tx.run(query)
    sentences = []
    answers = []
    for record in result:
        sentence = {
            's_id': record['s_id'],
            'sentence': record['sentence'],
            'topic': record['topic'],
            # 'topics': record['topics'],
            'sent_stemmed_overlap': record['sent_stemmed_overlap'],
            'sent_text': record['sent_text'],
            'sent_tags': record['sent_tags'],
            'sent_tokens': record['sent_tokens'],
            'nbr_text': record['nbr_text'],
            'nbr_tags': record['nbr_tags'],
            'nbr_tokens': record['nbr_tokens'],
            'topic1': record['topic1'],
            'topic2': record['topic2'],
            'answer_type': record['answer_type']
        }
        if '.' not in sentence['sentence'][-2:]:
            sentence['sentence'] += '.'
        sentences.append(sentence)

    for sentence in sentences:
        answers.append([sentence['sent_stemmed_overlap'], sentence['sent_text'], sentence['sent_tags'], sentence['sent_tokens'], sentence['nbr_text'], sentence['nbr_tags'], sentence['nbr_tokens'], sentence['topic1'], sentence['topic2'], sentence['answer_type']])

    answers = np.array(answers)
    mean = answers.mean(axis = 0).reshape((1, 10))
    std = answers.std(axis = 0)
    if 0 not in std:
        answers = (answers - mean)/std

    for i in range(len(sentences)):
        sentences[i]['score'] = sum(answers[i][j] * weights[j] for j in range(len(answers[i])))

    sentences.sort(key=lambda sentence: sentence['score'], reverse = True)
    return sentences[:top_k]

def get_topics_of_sentences(sentences):
    topics = {}
    for sentence in sentences:
        if sentence['topic'] not in topics:
            topics[sentence['topic']] = sentence['score']
        topics[sentence['topic']] = max(topics[sentence['topic']], sentence['score'])
    topic_list = [t[1] for t in sorted([(topics[t], t) for t in topics], reverse = True)]
    return topic_list

def get_top_topics():
    query = ('match (t:Topic)<-[]*-(p:Paragraph) return t.text as topic order by count(*) desc')

def get_neighbouring_sentences(sid):
    global driver
    with driver.session() as session:
        sentences = session.read_transaction(q_get_neighbouring_sentences, sid)
    return sentences

def q_get_neighbouring_sentences(tx, sid):
    query = ('match (s:Sentence) where abs(s.id - ' + str(sid) + ') <= 2 return s.id as s_id, s.text as sentence')
    records = tx.run(query)
    sentences = []
    for record in records:
        sentences.append({
            's_id': record['s_id'],
            'sentence': record['sentence']
            })
    return sentences




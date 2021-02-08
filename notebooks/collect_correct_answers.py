import time
import numpy as np

from helper import *

from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('chatbot', 'password'))

def shortlist_sentences(query):
	global driver

	start = time.time()

	with driver.session() as session:
		sentences = session.read_transaction(q_shortlist_sentences, query, 100)

	return sentences

def q_shortlist_sentences(tx, query, top_k):
    keywords = find_keywords(query)
    print(keywords)
    sent_weightage = 0.90985828
    para_weightage = 0.76878578
    topic1_weightage = 0.28471501
    topic2_weightage = 0.28471501
    query = (
        'with ' + str(keywords) + ' as keywords \n' +
        'match (main_topic:Topic)<-[]-(p:Paragraph)-[]->(s:Sentence)-[*]->(sent_e:ExtEntity) \n' + 
        'match (main_topic)<-[*]-(:Paragraph)-->(nbr_s:Sentence) \n' +
        'where abs(nbr_s.id - s.id) <= 2 \n' +
        'match (nbr_s)-[*]->(nbr_e:ExtEntity) \n' + 
        'match path = (t:Topic)<-[:about_topic*1..2]-(p) \n' +
        'with collect(distinct sent_e) as entities, collect(distinct nbr_e) as nbr_entities, collect(distinct [t, length(path)]) as topics, main_topic, s, keywords \n' + 
        'with main_topic.text as topic, topics, s.id as s_id, s.text as sentence, reduce( \n' + 
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'isPresent = 0.0, entity in entities | \n' + 
                'case \n' + 
                    'when entity.text = keyword[0] and isPresent < 1.0 then 1.0 \n' + 
                    'when entity.text starts with keyword[1] and isPresent < 0.7 then 0.7 \n' +
                    'else isPresent \n' + 
                'end \n' + 
            ') \n' + 
        ') as score_sent, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'isPresent = 0.0, entity in nbr_entities | \n' + 
                'case \n' + 
                    'when entity in entities then isPresent \n' + 
                    'when entity.text = keyword[0] and isPresent < 1 then 1 \n' + 
                    'when entity.text starts with keyword[1] and isPresent < 0.7 then 0.7 \n' + 
                    'else isPresent \n' + 
                'end \n' + 
            ') \n' +
        ') as score_nbr_sent, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
                'topics_matched = 0.0, topic in topics | \n' +
                'topics_matched + reduce( \n' + 
	                'isPresent = 0.0, entity in topic[0].keywords | \n' + 
	                'case \n' + 
	                    'when topic[1] = 1 and entity = keyword[0] and isPresent < 1.0 then 1.0 \n' + 
	                    'when topic[1] = 1 and entity starts with keyword[1] and isPresent < 0.7 then 0.7 \n' +
	                    'else isPresent \n' + 
	                'end \n' + 
                ') \n' +
            ') \n' + 
        ') as score_topic1, reduce( \n' +
            'total = 0.0, keyword in keywords| \n' + 
            'total + reduce( \n' + 
               'topics_matched = 0.0, topic in topics | \n' +
                'topics_matched + reduce( \n' + 
	                'isPresent = 0.0, entity in topic[0].keywords | \n' + 
	                'case \n' + 
	                    'when topic[1] = 1 and entity = keyword[0] and isPresent < 1.0 then 1.0 \n' + 
	                    'when topic[1] = 1 and entity starts with keyword[1] and isPresent < 0.7 then 0.7 \n' +
	                    'else isPresent \n' + 
	                'end \n' + 
                ') \n' +
            ') \n' + 
        ') as score_topic2 \n' +
        'with s_id, sentence, topic, topics, score_sent, score_nbr_sent, score_topic1, score_topic2, (' + str(sent_weightage) + ' * score_sent + ' + str(para_weightage) + ' * score_nbr_sent + ' + str(topic1_weightage) + ' * score_topic1 + ' + str(topic2_weightage) + ' * score_topic2) as score \n' + 
        'where score >= 1.0 \n' + 
        'return s_id, sentence, topic, topics, score_sent, score_nbr_sent, score_topic1, score_topic2, score \n' + 
        'order by score desc '
    )
    # print(query)
    result = tx.run(query)
    sentences = []
    for record in result:
        sentence = {
            's_id': record['s_id'],
            'sentence': record['sentence'],
            'topic': record['topic'],
            # 'topics': record['topics'],
            'score_sent': record['score_sent'],
            'score_nbr_sent': record['score_nbr_sent'],
            'score_topic1': record['score_topic1'],
            'score_topic2': record['score_topic2'],
            'score': record['score'],
        }
        if '.' not in sentence['sentence'][-2:]:
            sentence['sentence'] += '.'
        # print('(', sentence['score'],':', round(sentence['score_sent'], 2), round(sentence['score_nbr_sent'], 2), round(sentence['score_topic1'], 2), round(sentence['score_topic2'], 2), ')') 
        # print(str(sentence['topics']))
        # print(sentence['sentence'])
            # print()
        sentences.append(sentence)
    return sentences[:top_k]

history = read_json('../data/history.json')

dataX = []
dataY = []

for question in history:
    print('-' * 60)
    print(question)
    print('-' * 60)

    answer = []
    sentences = shortlist_sentences(question)

    answers = []
    for s in range(len(sentences)):
    	print(s, ':', sentences[s]['sentence'])
    	answers.append([sentences[s]['score_sent'], sentences[s]['score_nbr_sent'], sentences[s]['score_topic1'], sentences[s]['score_topic2']])

    answers = np.array(answers)
    mean = answers.mean(axis = 0).reshape((1, 4))
    std = answers.std(axis = 0)
    if 0 not in std:
       answers = (answers - mean)/std
    else:
        continue

    correct_answers = set(map(int, input().split()))

    for s in range(len(sentences)):
    	dataX.append([question, sentences[s]['sentence']] + [answers[s][i] for i in range(4)])
    	if s in correct_answers:
    		dataY.append(1)
    	else:
    		dataY.append(0)

    write_json({
    'X': dataX,
    'Y': dataY
    }, 'question_answer_pairs.json')

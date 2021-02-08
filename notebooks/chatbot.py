import time

from helper import *

from neo4j import GraphDatabase
driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('chatbot', 'password'))

from allennlp.predictors.predictor import Predictor
import allennlp_models.rc
mrc = Predictor.from_path("../bidaf-elmo-model-2020.03.19.tar.gz")


def shortlist_sentences(query):
	global driver

	start = time.time()

	with driver.session() as session:
		sentences = session.read_transaction(q_shortlist_sentences, query, 10)

	return sentences

def find_answer_from_mrc(question, sentences):
	global mrc

	passage = '\n'.join(sentence['topic'] + ': ' + sentence['sentence'] for sentence in sentences)
	answers = mrc.predict(question, passage)["best_span_str"].split('.')
	final_answer = []
	for answer in answers:
		for sentence in sentences:
			if answer in sentence['topic'] + ': ' + sentence['sentence']:
				# print(sentence['sentence'])
				final_answer.append(sentence)

	return final_answer

def q_shortlist_sentences(tx, query, top_k):
    keywords = find_keywords(query)
    print(keywords)
    sent_weightage, para_weightage, topic1_weightage, topic2_weightage = [1.17610199, 1.30961266, 0.31559684, 0.31559684]
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

history = read_json('../data/history.json')

data = read_json('question_answer_pairs.json')
clf = LogisticRegression(class_weight = {0: 1, 1: 100})
X = np.array([data['X'][i][2:] for i in range(len(data['X']))])
Y = np.array(data['Y'])
clf.fit(X, Y)


while True:
	print('-' * 60)
	print('Hello! Please ask your question!')

	print('-' * 60)
	question = input()

	answer = []
	sentences = shortlist_sentences(question)

	if len(sentences) == 0:
		print('-' * 60)
		print('Sorry, seems like we didn\'t find anything related to this question!')
		history[question] = None

	elif len(sentences) <= 3:
			answer = sentences

	if not answer:
		topics = get_topics_of_sentences(sentences)
		print('-' * 60)
		print('Which of the following is this question related to?')
		for i in range(len(topics)):
			print(i + 1, ':', topics[i])

		print('-' * 60)
		topic_list = map(int, input().split())

		topic_list = [topics[i - 1] for i in topic_list]

		if topic_list:
			sentences = [sentence for sentence in sentences if sentence['topic'] in topic_list]

		if len(sentences) <= 3:
			answer = sentences

	if not answer:
		print('-' * 60)
		print('Hmm...just wait for a moment, let me check...')
		answer = find_answer_from_mrc(question, sentences)
	
	print('-' * 60)
	for i in range(len(answer)):
		print(i + 1, answer[i]['sentence'])

	print('-' * 60)
	print('To expand more on any of these sentences, please enter sentence number or 0 otherwise')
	
	print('-' * 60)
	s_no = int(input())
	if s_no != 0:
		nbr_sents = get_neighbouring_sentences(answer[s_no - 1]['s_id'])
		print('-' * 60)
		for sentence in nbr_sents:
			print(sentence['sentence'])

	print('-' * 60)
	print('Was your question answered? [y/n]')

	print('-' * 60)
	isCorrect = input().lower() == 'y'

	history[question] = {
		'answer': answer,
		'correct': isCorrect
	}

	if isCorrect:
		print('-' * 60)
		print('Thank You!')
	else:
		print('-' * 60)
		print('Please mail admin department, sorry!')

	write_json(history, '../data/history.json')





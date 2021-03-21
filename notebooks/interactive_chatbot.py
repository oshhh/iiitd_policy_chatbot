from qa_helper import *

init_kg()
init_mrc()
# init_bert()

history = read_json('../data/history.json')

while True:
    print('-' * 60)
    print('Hello! Please ask your question!')

    print('-' * 60)
    question = input()
    # print(question)

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
        print('Please wait...')
        # answer = find_answer_from_bert(question, sentences)
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



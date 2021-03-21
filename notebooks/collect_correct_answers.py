from qa_helper import *
init_kg()


history = read_json('../data/history.json')

data = []

for question in history:
	print('-' * 60)
	print(question)
	print('-' * 60)

	answer = []
	sentences = shortlist_sentences(question)

	answers = []
	for s in range(len(sentences)):
		answers.append([sentences[s]['sent_stemmed_overlap'], sentences[s]['sent_text'], sentences[s]['sent_tags'], sentences[s]['sent_tokens'], sentences[s]['nbr_text'], sentences[s]['nbr_tags'], sentences[s]['nbr_tokens'], sentences[s]['topic1'], sentences[s]['topic2'], sentences[s]['answer_type']])
# 		print(s, ':', [round(val, 2) for val in answers[-1]], ':', sentences[s]['sentence'])
		print(s, ':', sentences[s]['sentence'])

	answers = np.array(answers)
	mean = answers.mean(axis = 0).reshape((1, 10))
	std = answers.std(axis = 0)
	for i in range(len(mean)):
		if std[i] != 0:
			answers[i, :] = (answers[i, :] - mean[i])/std[i]
		else:
			answers[i, :] = answers[i, :] - mean[i]

			
	correct_answers = set(map(int, input().split()))

	for s in range(len(sentences)):
		data.append([question, sentences[s]['sentence'], s in correct_answers] + [answers[s][i] for i in range(len(answers[s]))])


data = [['question', 'candidate answer', 'is correct', 'sent_stemmed_overlap', 'sent_text', 'sent_tags', 'sent_tokens', 'nbr_text', 'nbr_tags', 'nbr_tokens', 'topic1', 'topic2', 'answer_type']] + data

# write_csv(data, '../data/kg/outputs_kg_v6.csv')
write_csv(data, '../data/question_answer_pairs.csv')

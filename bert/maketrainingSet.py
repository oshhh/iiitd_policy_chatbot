import json

# def read_json(filename):
#     with open(filename) as file:
#         data = json.load(file)
#     return data

def write_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

def getTokenizedAnswers(context, answers):
	start_word = answers[:2]
	# end_word = answers[-2:]
	start_answer = context.index(answers)
	end_answer = start_answer + len(answers)
	return {
		"answer": answers,
		"start_answer": start_answer,
		"end_answer": end_answer
	}

data = dict()
pid = int(input())
while True:
	context = input()
	if context == "stop":
		break
	data[pid] = {
	'qas':{},
	'context': context
	}
	qid = 0
	qa = data[pid]["qas"]
	while True:
		questions = input()
		if questions == "stop":
			break
		qa[qid] = dict()
		answers = input()
		answers = getTokenizedAnswers(context, answers)
		qa[qid]["question"] = questions
		qa[qid]["answer"] = answers
		qid += 1
	pid += 1

# date_old = read_json("training_set.json")
# date_old.update(data)
write_json(data, "training_set.json")
import argparse
import json, os

if __name__ == '__main__':
    save_list = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=str)
    args = parser.parse_args()

    responses = json.load(open(args.evaluation, 'r'))
    assert(len(responses) == 96)

    # analyze responses
    num_0 = 0
    scores = []
    for i, response in enumerate(responses):
        response = response['choices'][0]['message']['content']
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
            save_list[str(i)] = scores_found[0]
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)
            num_0 = num_0+1

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join([str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]), flush=True)
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
args = parser.parse_args()

with open(args.evaluation, 'r') as f:
    datas = json.load(f)

cumu_scores = []
mean_scores = []
for data in datas:

    ratings = data["rating"]
    labels = data['label']

    # compute all metrics
    cumu_scores.extend(ratings)
    mean_scores.append(sum(ratings)/len(ratings))

cumu_hallucination, mean_hallucination = [], []
for s in cumu_scores:
    if s >= 3:
        cumu_hallucination.append(0)
    else:
        cumu_hallucination.append(1)

for s in mean_scores:
    if s >= 3:
        mean_hallucination.append(0)
    else:
        mean_hallucination.append(1)

print('Cumulative Average score: {:.2f}'.format(sum(cumu_scores) / len(cumu_scores)))
print('Cumulative Hallucination rate: {:.2f}'.format(sum(cumu_hallucination) / len(cumu_hallucination)))
print('Mean Average score: {:.2f}'.format(sum(mean_scores) / len(mean_scores)))
print('Mean Hallucination rate: {:.2f}'.format(sum(mean_hallucination) / len(mean_hallucination)))
print("-------------")

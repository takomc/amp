import openai
import argparse
import json
import time
from tqdm import tqdm
import re



template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Rounds of Conversations: 2
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Question: What color is the left cat?
- Standard Human-Generated Answer: The color of the left cat is orange and white.
- LMM Response to Evaluate: The left cat is orange and white.
- Analysis: In the first round, the LMM's response is false because it claims there is only one cat in the image, when in fact there are two. In the second round, the LMM's response is correct because it accurately identifies the color of the cat.
- Hallucination: [Yes, No]

### Example 2:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Rounds of Conversations: 4
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Question: Are there any plants in this image?
- Standard Human-Generated Answer: No, the image does not contain any plants.
- LMM Response to Evaluate: Yes, there are some plants in front of the building.
- Question: How many bags are in this picture?
- Standard Human-Generated Answer: There are three bags in total, two on the ground and one in the man's hand.
- LMM Response to Evaluate: There are two bags in this picture. They are on the ground.
- Question: What is the man doing?
- Standard Human-Generated Answer: The man is taking a photo with a big smile and a scissors hand gesture.
- LMM Response to Evaluate: The man is taking photos at the airport. He is very happy from the smile on his face.
- Analysis: In the first round, although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer. In the second round, the LMM's response is wrong because it claims the existence of non-existent plants. In the third round, although the LMM predicts the right number of bags. However, it provides the wrong location of the bags. In the fourth round, the LMM's output is correct, but it isn't a perfect response because it omits the scissors hand gesture.
- Hallucination: [No, Yes, Yes, No]

### Example 3:
- Image Contents: Tree, Plant, Panda, Animal
- Rounds of Conversations: 3
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo, we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Question: Can you infer the weather in this image?
- Standard Human-Generated Answer: The weather is rainy, as indicated by the falling raindrops and the fact that people are holding umbrellas.
- LMM Response to Evaluate: From the visible falling raindrops, the weather in this image is rainy.
- Question: Describe the positions of animals and trees.
- Standard Human-Generated Answer: The animals are in front of trees.
- LMM Response to Evaluate: The animals are surrounded by the trees.
- Analysis: In the first round, the LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo. In the second round, the LMM's response is right, but it does not give the perfect reasons. For example, it neglects the people with umbrellas. In the third round, the LLM's response is wrong because it claims animals are surrounded by trees, but they are in front of the trees.
- Hallucination: [No, No, Yes]

### Example 4:
- Image Contents: Tree, Tower
- Rounds of Conversations: 1
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: In the first round, although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: [No]

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evaluation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response for each round of conversation by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

Note that the ratings for responses must follow the list format. For example, the conversations contain four rounds, and the ratings for responses are "6,3,0,1". Then the ratings you output is "Ratings: [6, 3, 0, 1]".

### Image Contents
{}

### Rounds of Conversations
{}
'''

res_template = """
### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
"""

parser = argparse.ArgumentParser()
parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
parser.add_argument('--api-key', type=str, required=True)
parser.add_argument('--gpt-model', type=str, default='gpt-4-0314')
args = parser.parse_args()

# load json file
openai.api_key = args.api_key
with open(args.response, 'r') as f:
    records = json.load(f)

print(len(records))

# ask GPT-4 to evaluate
responses = []
for i, record in enumerate(records):
    save_dict = {}
    save_dict['id'] = record['id']
    save_dict['image'] = record['image']
    save_dict['label'] = record['label']
    save_dict['rounds'] = len(record['question'])
    image_content = ', '.join(record['image_content'])
    input_text = template.format(image_content, str(len(record['question'])))
    for j in range(len(record['question'])):
        input_text = input_text+res_template.format(record['question'][j].replace('<image>\n', '').strip(), record['gt_answer'][j].strip(), record['model_answer'][j].strip())

    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-32k-0613",
                model=args.gpt_model,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=0.0,
            )
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue

    print(i, response['choices'][0]['message']['content'], flush=True)
    save_dict['response'] = response['choices'][0]['message']['content']
    responses.append(save_dict)
    time.sleep(1)

# analyze responses
scores = []
final_list = []
for i, response in enumerate(responses):

    response['rating'] = []
    rounds = response['rounds']
    response_str = response['response']

    # Find Rating List
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, response_str)
    if len(matches) == 1:
        match_item = matches[0].split(',')
    elif len(matches) == 2:
        for match in matches:
            if "Yes" not in match and "No" not in match:
                match_item = match.split(',')
    assert len(match_item) == rounds

    for match in match_item:
        if "N/A" in match:
            match = 0
        scores.append(int(match))
        response['rating'].append(int(match))

    final_list.append(response)

hallucination = []
for s in scores:
    if s >= 3:
        hallucination.append(0)
    else:
        hallucination.append(1)

if args.evaluation is not None:
    with open(args.evaluation, 'w') as f:
        json.dump(final_list, f, indent=2)
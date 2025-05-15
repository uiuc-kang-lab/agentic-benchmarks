system_prompt = "You are an expert in understanding the semantical differences and similarities between two sentences."

task_prompt = """
In this task, you will be given a question, two answers, and an annotation on whether the answers match. Your task is to judge whether the annotation is correct and provide an explanation. Here are some examples:

Example 1
Question: What are the key aspects that the customers don't like about Circe ice fleece?
Answer 1: The key aspects that customers don't like about the Circe Hooded Ice Fleece are: poor material quality that pills after washing, strange fit with short sleeves and a baggy body, lack of warmth for winter walks, and an oversized hood that falls over the face. Customers suggest there are better options available for the price.
Answer 2: Material quality, fit, insufficient warmth, color
Annotation: Match
Judgement: Wrong annotation
Explanation: Answer 1 does not mention color, while answer 2 mentions color. Therefore, the two answers do not match. The annotation is wrong.

Example 2
Question: Give me the product names and the sizes of the products that have 2-3 units left.
Answer 1: There are no products with 2-3 units left in stock according to the current report. Therefore, there are no product names or sizes to provide.
Answer 2: Eos V-Neck Hoodie: S Minera Luma Tech V-Tee: XS
Annotation: Match
Judgement: Wrong annotation
Explanation: Answer 1 does not mention the two products mentioned in Answer 2. Therefore, the two answers do not match. The annotation is wrong.

Example 3
Question: Tell me the full names of the repositories where I made contributions and they got the least stars.	
Answer 1: The repositories where you have contributed and that have received the fewest stars (0 stars) are: 1. Byte Blaze / chatgpt_plugin 2. Byte Blaze / cloud-to-butt 3. Byte Blaze / dotfiles 4. Byte Blaze / gimmiethat.space 5. Byte Blaze / remove-board-movement-events-from-the-github-issue-timeline 6. Byte Blaze / solarized-prism-theme 7. Byte Blaze / timeit 8. Byte Blaze / web_agent_index 9. x-lab / web_arena
Answer 2: cloud-to-butt, dotfiles, timeit, solarized-prism-theme, gimmiethat.space, remove-board-movement-events-from-the-github-issue-timeline
Annotation: Match
Judgement: Wrong annotation
Explanation: Answer 1 includes 3 more repositories: chatgpt_plugin, web_agent_index, web_arena. Therefore, the two answers do not match. The annotation is wrong.

Example 4
Question: How many commits did Steven Woodson make to a11y-webring.club on 2/6/2023?
Answer 1: 3
Answer 2: 5
Annotation: Match
Judgement:	Wrong annotation
Explanation: Answer 1 and Answer 2 show different numbers. Therefore, the two answers do not match. The annotation is wrong.

Example 5
Question: Show me the customers who have expressed dissatisfaction with Olivia zip jacket?
Answer 1: The list of customers who have given negative feedback about the Olivia 1/4 Zip Light Jacket includes: 1. Nickname: Emma, Review: "I was really disappointed with the Olivia 1/4 Z..." 2. Nickname: seam miller, Review: "I have to say, I'm not impressed with the Olivi...".
Answer 2: Emma Lopez, Seam Miller
Annotation: Not match
Judgement: Wrong annotation
Explanation: Both Answer 1 and Answer 2 mention 2 customers: Emma and Seam. Therefore, the two answers do not match. The annotation is wrong.

Example 6
Question: Where is the nearest Starbucks to Carnegie Mellon, and what is the walking distance to it?
Answer 1: Starbucks, 417, South Craig Street, Bellefield, Pittsburgh, Allegheny County, Pennsylvania, 15213, United States, 557m
Answer 2: The nearest Starbucks to Carnegie Mellon University is located at 417 South Craig Street, Bellefield, Pittsburgh, Allegheny County, 15213, United States, with a walking distance of 557 meters.
Annotation: Not match
Judgement: Wrong annotation
Explanation: Both Answer 1 and Answer 2 show the same address with the same distance. Therefore, the two answers match. The annotation is wrong.

Example 7
Question: What is the zip code of Chatham University?
Answer 1: 15232, 15208	
Answer 2: 15232
Annotation: Match
Judgement: Wrong Annotation
Explanation: Answer 1 mentions 2 zip codes while Answer 2 mentions 1 zip code. Therefore, the two answers do not match. The annotation is wrong.

Example 8
Question: Which customer(s) has completed the second most number of orders in the entire history?	
Answer 1: The customer(s) with the second most number of orders in the entire history from 2020 to 2025 are John Smith, Jane Doe, Bob Johnson, Matt Baker, Grace Nguyen, Lucy Garcia, Ava Brown, Adam Garcia, Sarah Miller, Grace Nguyen, and Michael Nguyen, each with 5 orders.
Answer 2: Adam Garcia, Michael Nguyen, Sarah Miller
Annotation: Match
Judgement: Wrong annotation
Explanation: Answer 1 shows 7 more customers than Answer 2. Therefore, the two answers match. The annotation is wrong.

Example 9
Question: What is the duration required to first walk from Massachusetts Institute of Technology to Harvard University, and then drive to Boston Logan International Airport?	
Answer 1: The duration required to first walk from Massachusetts Institute of Technology to Harvard University is 45 minutes, and then drive to Boston Logan International Airport is 8 minutes.
Answer 2: 63 min
Annotation: Match
Judgement: Wrong annotation
Explanation: Answer 1 shows 45+8=53 minutes, while Answer 2 shows 63 minutes. Therefore, the two answers do not match. The annotation is wrong.

Example 10
Question: What is the top-1 best-selling product in 2022?
Answer 1: Quest Lumaflex Band
Answer 2: Quest Lumaflex Band
Annotation: Match
Judgement: Correct annotation
Explanation: Both answers show the same product. Therefore, the two answers match. The annotation is correct.

Example 11
Question: What is the top-1 best-selling product type in Quarter 1 2022?
Answer 1: Yoga ball
Answer 2: The best-selling products for the first quarter of 2022 are the 'Dash Digital Watch' and 'Quest Lumaflex™ Band', both with an order quantity of 3.
Annotation: Not match
Judgement: Correct annotation
Explanation: Answer 2 shows two different products compared the product in Answer 1. Therefore, the two answers do not match. The annotation is correct.

Here is the question, its two answers, and the annotation:
Question: __placeholder1__
Answer 1: __placeholder2__
Answer 2: __placeholder5__
Annotation: __placeholder4__
Please judge if the annotation is correct or incorrect and explain your judgment.
"""


from openai import OpenAI
import json 
import argparse
from pydantic import BaseModel

class AnnotationJudgement(BaseModel):
    Judgement: str
    Explanation: str
    

args = argparse.ArgumentParser()
args.add_argument('--agent', type=str, default='step')
args = args.parse_args()

agent_name = args.agent
with open(f'./agents/{agent_name}/annotation.json', 'r') as f:
  test_set = json.load(f)

client = OpenAI()

input_tokens = 0
output_tokens = 0

for item in test_set:
  new_task_prompt = task_prompt.replace('__placeholder1__', item['question'])
  new_task_prompt = new_task_prompt.replace('__placeholder2__', item['agent_answer']).replace('__placeholder4__', item['anno'])
  new_task_prompt = new_task_prompt.replace('__placeholder5__', str(item['eval']))
  completion = client.beta.chat.completions.parse(
      model="o1",
      messages=[
          {
              "role": "system",
              "content": system_prompt
          },
          {
            "role":"user",
            "content": new_task_prompt
          }
      ],
     response_format=AnnotationJudgement
  )

  result = completion.choices[0].message.parsed
  output_dict = {'task_id': item['task_id'], 'Judgement': result.Judgement, 'Explanation': result.Explanation}
  with open(f"{agent_name}_o1_results.jsonl", "a") as f:
    f.write(json.dumps(output_dict) + "\n")
  input_tokens += completion.usage.prompt_tokens
  output_tokens += completion.usage.completion_tokens


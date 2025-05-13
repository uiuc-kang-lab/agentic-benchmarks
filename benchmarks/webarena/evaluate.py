import json 
import re
import argparse

args = argparse.ArgumentParser()
args.add_argument('--agent', type=str, default='agentoccam')
args.add_argument('--mode', type=str, default='original')
args = args.parse_args()

agent_name = args.agent
mode = args.mode 

if agent_name.lower() == 'agentsymbiotic':
  agent_file = './agents/AgentSymbiotic/webarena'
elif agent_name.lower() == 'agentoccam':
  agent_file = './agents/agentoccam/agentoccam-judge'
elif agent_name.lower() == 'step':
  agent_file = './agents/step/webarena'

if mode == 'original':
  from evaluation_harness import evaluator_router, StringEvaluator
  config_path = 'config_files'
else:
  from evaluation import evaluator_router, StringEvaluator
  config_path = 'config'


res = []
count = 0
success = 0
grader_success = 0
for i in range(812):
    with open(f"./{config_path}/{i}.json", "r") as f:
      data = json.load(f)
  
    eval_type = data['eval']['eval_types']
    if 'string_match' in eval_type and len(eval_type) == 1:
      if 'fuzzy_match' in data['eval']['reference_answers']:
        continue

      with open(f'{agent_file}/{i}.json','r') as f:
          test_data = json.load(f)
      tra = test_data['trajectory']
      label_score = tra[-1]['success']
      last_action = tra[-1]['action']   

      if label_score == 1:
        success += 1
      match = re.search(r"stop ?\[(.+)\]", last_action, re.DOTALL)

      if not match:
          answer = ""
      else:
          answer = match.group(1)
          if answer[-1] == ']':
            answer = answer[:-1]
      trajectory = [{ "answer": answer}]

      eva = StringEvaluator()
      score = eva(trajectory=trajectory, config_file=f"./{config_path}/{i}.json", )

      if score == 1:
        grader_success += 1
      if (score == 1 and label_score < 1) or (score < 1 and label_score == 1):
        res.append((i,score,label_score,  answer, data['eval']['reference_answers']))

      count += 1


with open(f'{agent_name}_{mode}.log', 'w') as f:
  f.write(f'The number of total tasks: {count}\n')
  f.write(f'The number of successful tasks in the trajectory: {success}\n')
  f.write(f'The number of successful tasks by grader: {grader_success}\n')
  f.write('The details of the tasks that are different:\n')
  for item in res:
    f.write(str(item))
    f.write('\n')

from datasets import load_dataset
import pandas as pd
import os
import numpy as np

np.random.seed(2333)

dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
level_1_data = dataset["level_1"].to_pandas()
level_1_data["id"] = level_1_data.index

# only conisder the correct samples
level_1_data = level_1_data[level_1_data["Correct"] == True]

# sample a subset of the data for additional test case generation
# total sample size: 1000 for level_1
# the sample size for each task is proportional to the number of correct samples of that task
level_1_data_by_task = level_1_data.groupby("Task_ID").size()
total_level_1 = level_1_data_by_task.sum()
level_1_sample_size = (level_1_data_by_task / total_level_1 * 1000).round().astype(int)
level_1_residue = 1000 - level_1_sample_size.sum()

level_1_sampled_data = pd.DataFrame()
for task, size in level_1_sample_size.items():
    task_data = level_1_data[level_1_data["Task_ID"] == task]
    sampled_data = task_data.sample(n=size)
    level_1_sampled_data = pd.concat([level_1_sampled_data, sampled_data])

if level_1_residue > 0:
    task_data = level_1_data[level_1_data["Task_ID"] == task]
    sampled_data = task_data.sample(n=level_1_residue)
    level_1_sampled_data = pd.concat([level_1_sampled_data, sampled_data])
print(level_1_sampled_data)

# save the sampled data
os.makedirs("data", exist_ok=True)
level_1_sampled_data.to_parquet("data/level_1_sampled_data.parquet", index=False)

# save kernels
level_1_count = 0
for index, row in level_1_data.iterrows():
    os.makedirs(f"kernels/level_1/{row['Op_Name']}", exist_ok=True)
    if not row['CUDA_Code'] or row['Correct'] == False:
        continue
    with open(f"kernels/level_1/{row['Op_Name']}/{row['id']}.cu", "w") as f:
        f.write(row["CUDA_Code"])
        level_1_count += 1
print(f"level_1_count: {level_1_count}")

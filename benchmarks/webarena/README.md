# Setup
You only need to replace 'openai_api_key' with your own key; the other fields (e.g., <your_shopping_site_domain>) should remain unchanged.
```bash
conda create -n webarena python=3.11
conda activate webarena
pip install -r requirement.txt
export OPENAI_API_KEY=<openai_api_key>

export SHOPPING="<your_shopping_site_domain>:7770"
export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export REDDIT="<your_reddit_domain>:9999"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"
export WIKIPEDIA="<your_wikipedia_domain>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="<your_homepage_domain>:4399" # this is a placeholder
```

# Download the agent trajectories
```bash
gdown 1yXr8LkFImvDaPW7TnDFI_12ZthVEn4oJ
```

# Evaluate the original grader
```bash
python evaluate.py --agent agent_name
```

# Evaluate the customized grader proposed by AgentOccam
```bash
python evaluate.py --agent agent_name --mode customized
```

# Evaluate the agent result by o1 
```bash
python llm_as_a_judge.py --agent agent_name
```
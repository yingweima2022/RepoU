
## Hi there 👋

**RepoU** is a ✨ Official implementation ✨ repository of paper "Alibaba LingmaAgent: Improving Automated Issue Resolution via Comprehensive Repository Exploration".

- Recently, Large Language Model (LLM) based agents have advanced the significant development of Automatic Software Engineering (ASE).
- We develop a novel ASE method named RepoUnderstander by guiding agents to comprehensively understand the whole repositories.
- It achieved 18.5% relative improvement on the SWE-bench Lite benchmark compared to SWE-agent.


![20240604225103](https://github.com/RepoUnderstander/RepoUnderstander/assets/170649488/7f9a862a-48b1-47d4-a287-dd4705a6d5d3)


1. Before running, you should make sure you have the SWE-bench environment installed.
```
See https://github.com/princeton-nlp/SWE-bench
```

2. Configure OpenAI token
```
RepoUnderstaner/app/model/gpt.py
```

3. Run our code:
```
python scripts/run.py conf/vanilla-lite.conf
```

4. MCTS Code
```
RepoUnderstaner/app/MCTS/
```

5. FIX DATASET
```
fix_dataset.jsonl

format (jsonl):
- instance_id
- signiture_changed_by_rule
- has_additional_info
- additional_info
- problem_statement
- patch
```
## Acknowledgements

Thanks for the following open-source projects:
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent)
- [AutoCodeRover](https://github.com/nus-apr/auto-code-rover)

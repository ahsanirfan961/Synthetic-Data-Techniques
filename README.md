# Synthetic Data Generation Techniques

This repository provides various techniques for synthetic data generation.

## Techniques
1. Self Instruct
2. Magpie
3. Agent Instruct with Arena Learning
4. Storm 

## How to Run
**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```
**2. Setup configs:**

Put your huggingface hub access token in [config.yaml](config.yaml)

The directory of each technique in [techniques](./techniques/) contains a config.yaml file. Adjust the values of datasets and models in that file.

**3. Run the main.py script:**
Specify the config.yaml file path and the technique name while runnning this command.
```bash
python main.py --config=config.yaml --technique=magpie
```
Example techniques are,
1. agent-arena
2. magpie
3. self-instruct
4. storm

For help run,
```bash
python main.py --help
```

## License
This repository is licensed under [MIT-License](MIT-LICENSE.txt).
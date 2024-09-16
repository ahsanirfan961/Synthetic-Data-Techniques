import argparse
import yaml
from techniques.agent_instruct.code import AgentInstructTechnique
from techniques.arena_learning.code import ArenaLearningTechnique
from techniques.bonito.code import BonitoTechnique
from techniques.genstruct.code import GenstructTechnique
from techniques.instruction_synthesizer.code import InstructionSynthesizerTechnique
from techniques.magpie.code import MagpieTechnique
from techniques.self_instruct.code import SelfInstructTechnique
from techniques.storm.code import StormTechnique
from techniques.sentence_similarity.code import SentenceSimilarityTechnique
from techniques.tiny_stories.code import TinyStoriesTechnique


help = '''Please specify a technique to run,
    e.g 1. agent-instruct
        2. magpie
        3. self-instruct
        4. storm-curation
        5. genstruct
        6. synthesizer
        7. arena-learning
        8. bonito
        9. sentence-similarity
        10. tiny-stories
    '''

# Step 1: Use argparse to parse the --config argument
parser = argparse.ArgumentParser(description="Process a YAML config file.")
parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
parser.add_argument('--technique', type=str, required=True, help=help)
args = parser.parse_args()

# Step 2: Load and parse the YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Logging into huggingface_hub

HF_AUTH_TOKEN=config['hf-token']
from huggingface_hub import login
login(token=HF_AUTH_TOKEN)

if args.technique == 'agent-instruct':
    AgentInstructTechnique(config).process()
elif args.technique == 'magpie':
    MagpieTechnique(config).process()
elif args.technique == 'self-instruct':
    SelfInstructTechnique(config).process()
elif args.technique == 'storm-curation':
    StormTechnique(config).process()
elif args.technique == 'genstruct':
    GenstructTechnique(config).process()
elif args.technique == 'synthesizer':
    InstructionSynthesizerTechnique(config).process()
elif args.technique == 'arena-learning':
    ArenaLearningTechnique(config).process()
elif args.technique == 'bonito':
    BonitoTechnique(config).process()
elif args.technique == 'sentence-similarity':
    SentenceSimilarityTechnique(config).process()
elif args.technique == 'tiny-stories':
    TinyStoriesTechnique(config).process()
else:
    print('Invalid technique name')
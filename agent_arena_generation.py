import argparse
import yaml
from datasets import DatasetDict, Dataset
import pandas as pd
from distilabel.llms import TransformersLLM,OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput, CombineColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct, UltraFeedback
from typing import List
from pydantic import Field

# Step 1: Use argparse to parse the --config argument
parser = argparse.ArgumentParser(description="Process a YAML config file.")
parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
args = parser.parse_args()

# Step 2: Load and parse the YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

criteria_for_query_generation = (
    "1. Relevance: Ensure the questions are directly related to the content and context of the input paragraph."
    "2. Diversity: Include a variety of question types such as factual, analytical, inferential, and evaluative."
    "3. Clarity: Make sure each question is clear, concise, and unambiguous."
    "4. Complexity: Incorporate questions of varying difficulty levels, from simple recall to complex analysis."
    "5. Coverage: Cover the entire content of the paragraph, addressing different sections and key points."
    "6. Specificity: Frame questions to be specific and pointed, encouraging precise answers."
    "7. Engagement: Create questions that are interesting and engaging, promoting thoughtful responses."
    "8. Open-endedness: A portion of the generated questions should encourage creative and thoughtful responses, rather than simple factual recall."
    "9. Output: Provide only the five user queries without any introductory or explanatory text."
)

question_prompt = config.get('question_gen_prompt')
suggest_prompt = config.get('suggestion_gen_prompt')
refined_question_prompt = config.get('refined_question_gen_prompt')

BATCH_SIZE = config['input-batch-size']

DATASETS = config['datasets']

# Logging into huggingface_hub

HF_AUTH_TOKEN=config['hf-token']
from huggingface_hub import login
login(token=HF_AUTH_TOKEN)


#################################
# Defining Useful Functions     #
#################################

class InstructionSplitter:
  def split_instructions_from_dataset(self, dataset: Dataset):
    new_rows = []
    for row in dataset:
      new_rows.extend(self.split_instructions_from_row(row))
    return new_rows

  def split_instructions_from_row(self, row):
      results = []
      for instruction in row['instructions']:
          result = row.copy()
          result['instruction'] = instruction
          del result['instructions']
          results.append(result)
      return results

class RenameColumn(Step):

    old_column: str = Field(..., description="The name of the column to rename.")
    new_column: str = Field(..., description="The new name for the column.")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return [self.old_column]

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return [self.new_column]

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
            if self.old_column in example:
                example[self.new_column] = example.pop(self.old_column)  # Rename the column
        yield inputs

class ReplaceAllColumnValues(Step):
    column_name: str = Field(..., description="The name of the column whose values will be changed.")
    new_value: str = Field(..., description="The new value that will replace all existing values in the column.")

    @property
    def inputs(self) -> List[str]:
        return [self.column_name]

    @property
    def outputs(self) -> List[str]:
        return [self.column_name]

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
            if self.column_name in example:
                example[self.column_name] = self.new_value  # Update the column value
        yield inputs

class SplitInstructions(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['instructions']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instruction']

    def process(self, inputs: StepInput) -> StepOutput:
        inputs = InstructionSplitter().split_instructions_from_dataset(inputs)
        yield inputs

class MergeQuestionSuggesions(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['question', 'suggestions']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instruction']

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
          combined_text = example['question'] + "\n\nSuggestions:\n" + example['suggestions']
          example['instruction'] = combined_text
        yield inputs

# Define the function to split the 'generation' column into multiple rows
def split_generation(examples):
    # Ensure examples is a dictionary with lists
    generations = examples['generation']

    # Process each entry in the batch
    new_examples = []
    for generation in generations:
        questions = generation.split('?')
        questions = [q.strip() + '?' for q in questions if q.strip()]
        for question in questions:
            new_example = {
                'instruction': question,
                **{k: v for k, v in examples.items() if k != 'generation'}
            }
            new_examples.append(new_example)

    return {'instruction': [e['instruction'] for e in new_examples]}

def clean_text(text):
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

def clean_generations(generations):
    return [clean_text(item) for item in generations]

class CleanGenerations(Step):

    @property
    def inputs(self) -> List[str]:
        return ['generations']

    @property
    def outputs(self) -> List[str]:
        return ['generations']

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
          input['generations'] = clean_generations(input['generations'])
        yield inputs

def select_best_generation(row):
    base_rating = row['ratings'][0]
    # print(base_rating)
    competitor_ratings = row['ratings'][1:]
    # print(competitor_ratings)
    base_answer = row['generations'][0]
    answers = row['generations'][1:]

    max_competitor_rating = max(competitor_ratings)
    # print(max_competitor_rating)
    if max_competitor_rating > base_rating:
        # print("in if")
        best_competitor_index = competitor_ratings.index(max_competitor_rating) + 1
        # print(best_competitor_index)
        row['generations'][0] = row['generations'][best_competitor_index]
        row['generations'][best_competitor_index] = base_answer
    row['generation'] = row['generations'][0]
    return row

class SelectBestGeneration(Step):

    @property
    def inputs(self) -> List[str]:
        return ['generations', 'ratings']

    @property
    def outputs(self) -> List[str]:
        return ['generation']

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
          input = select_best_generation(input)
        yield inputs


##################################
# Creating and Running Pipelines #
##################################


##################################
#         Agent Instruct         #
##################################

models = config['models']

shared_model = TransformersLLM(model=models['agent-instruct'], device="cuda:0")

with Pipeline(name="Question Generation") as pipeline:
    load_hub_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"}
    )

    text_generation = TextGeneration(
        # llm = TransformersLLM(model="microsoft/Phi-3-mini-4k-instruct"),
        # llm = TransformersLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", device= "cuda:0"),
        llm = shared_model,
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
        output_mappings={"generation": "input", "model_name": "transformed_text_model"},
    )

    self_instruct = SelfInstruct(
        llm = shared_model,
        # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
        num_instructions=5,
        criteria_for_query_generation=criteria_for_query_generation,
        application_description=question_prompt,
        output_mappings={"model_name": "instructions_model"},
    )

    rename_1 = RenameColumn(
        name="rename_instr_to_raw_seed",
        old_column="instruction",
        new_column="raw_seed"
    )

    split_instr = SplitInstructions(
        name="split_instructions_step"
    )

    prompt_change_1 = ReplaceAllColumnValues(
        name="suggestion_system_prompt",
        column_name="system_prompt",
        new_value=suggest_prompt
    )

    suggestion_generation = TextGeneration(
        # llm = TransformersLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", device= "cuda:0"),
        llm = shared_model,
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
        output_mappings={"generation": "suggestions", "model_name": "suggestions_model"},
    )

    rename_2 = RenameColumn(
        name="rename_instr_to_question",
        old_column="instruction",
        new_column="question"
    )

    merge_question_suggestions = MergeQuestionSuggesions(
        name="merge_question_suggestions_step"
    )

    prompt_change_2 = ReplaceAllColumnValues(
        name="question_system_prompt",
        column_name="system_prompt",
        new_value=refined_question_prompt
    )

    question_generation = TextGeneration(
        llm = shared_model,
        # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
        output_mappings={"model_name": "refined_q_model"},
    )

    keep_columns = KeepColumns(
        columns=["generation"],
    )

    load_hub_dataset >> text_generation >> self_instruct >> rename_1 >> split_instr >> prompt_change_1 >> suggestion_generation >> rename_2 >> merge_question_suggestions >> prompt_change_2 >> question_generation >> keep_columns


distiset = pipeline.run(
    parameters={
        load_hub_dataset.name: {
            "repo_id": DATASETS['initial'],
            "split": "train",
        },
        text_generation.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": config['max-tokens'],
                    "temperature": config['temperature'],
                },
            },
        },
        self_instruct.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": config['max-tokens'],
                    "temperature": config['temperature'],
                },
            },
        },
        suggestion_generation.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": config['max-tokens'],
                    "temperature": config['temperature'],
                },
            },
        },
        question_generation.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": config['max-tokens'],
                    "temperature": config['temperature'],
                },
            },
        },
    },
)

# Map the function to split 'generation' column
split_dataset = distiset['default']['train'].map(
    split_generation,
    batched=True,
    remove_columns=['generation'],
    batch_size=1
)


# Pushing dataset after question generation to huggingface
split_dataset.push_to_hub(
    DATASETS['instructions-push'],
    token = HF_AUTH_TOKEN
)


##################################
#         Arena Learning         #
##################################

models:dict = config['models']['arena-learning']

with Pipeline(name="Battle of LLMs") as pipeline:
    load_dataset = LoadDataFromHub(
        name="dataset_for_arena_learning",
    )

    # first answer
    text_generation_1 = TextGeneration(
        name = "text_generation_01",
        llm = TransformersLLM(model=models['answer-gen'][0], device= "cuda:0"),
        # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
    )

    # second answer
    text_generation_2 = TextGeneration(
        name = "text_generation_02",
        # llm = TransformersLLM(model="mistralai/Mistral-7B-v0.1", device= "cuda:0"),
        llm = TransformersLLM(model=models['answer-gen'][1], device= "cuda:0"),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
    )

    # third answer
    text_generation_3 = TextGeneration(
        name = "text_generation_04",
        llm = TransformersLLM(model=models['answer-gen'][2], device= "cuda:0"),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
    )

    combine_columns = CombineColumns(
        name="combine_columns",
        columns=["generation", "model_name"],
        output_columns=["generations", "model_name"],
        input_batch_size=BATCH_SIZE
    )

    keep_columns_1 = KeepColumns(
        columns = ["instruction", "generations"]
    )

    clean = CleanGenerations(
        name="clean_generations"
    )

    ultrafeedback = UltraFeedback(
        llm=OpenAILLM(model=models['ranking'], api_key=config['gpt4-key']),
        input_batch_size=BATCH_SIZE,
        add_raw_output=False,
        aspect="overall-rating",
        output_mappings={"model_name": "ultrafeedback_model"}
    )

    best_gen = SelectBestGeneration(
        name="select_best_gen"
    )

    keep_columns = KeepColumns(
        columns=["instruction", "generation"]
    )

    load_dataset >> [text_generation_1, text_generation_2, text_generation_3] >> combine_columns >> keep_columns_1 >> clean >> ultrafeedback >> best_gen >> keep_columns


distiset_1 = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": DATASETS['instructions-push'],
                "split": "train",
            },
            text_generation_1.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": config['temperature'],
                        "max_new_tokens": config['max-tokens'],
                    }
                }
            },
            text_generation_2.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": config['temperature'],
                        "max_new_tokens": config['max-tokens'],
                    }
                }
            },
            text_generation_3.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": config['temperature'],
                        "max_new_tokens": config['max-tokens'],
                    }
                }
            },
            ultrafeedback.name: {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": config['max-tokens'],
                        "temperature": config['temperature'],
                    }
                }
            },
        },
    )


distiset_1.push_to_hub(
    DATASETS['intruction-response-push'],
    token = HF_AUTH_TOKEN
)

from distilabel.llms import TransformersLLM,OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput, CombineColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct, UltraFeedback
from typing import List
from pydantic import Field
import yaml


##################################
#         Arena Learning         #
##################################

class ArenaLearningTechnique:
    
    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        response_models_config = [model for model in config['models'] if model['type'] == 'response']
        ranking_model_config = next((model for model in config['models'] if model['type'] == 'rank'), None)

        for model_config in response_models_config:
            if model_config['vendor'] == 'openai':
                model = OpenAILLM(model=model_config['path'], api_key=model_config['api_key'])
            elif model_config['vendor'] == 'huggingface':
                model = TransformersLLM(model=model_config['path'], device="cuda:0")
            self.response_models.append(model)

        if ranking_model_config['vendor'] == 'openai':
            self.ranking_model = OpenAILLM(model=ranking_model_config['path'], api_key=ranking_model_config['api_key'])
        elif ranking_model_config['vendor'] == 'huggingface':
            self.ranking_model = TransformersLLM(model=ranking_model_config['path'], device="cuda:0")

        BATCH_SIZE = config['input-batch-size']

        with Pipeline(name="Battle of LLMs") as self.pipeline:
            self.load_dataset = LoadDataFromHub(
                name="dataset_for_arena_learning",
            )

            # first answer
            self.text_generation_1 = TextGeneration(
                name = "text_generation_01",
                llm = self.response_models[0],
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            # second answer
            self.text_generation_2 = TextGeneration(
                name = "text_generation_02",
                llm = self.response_models[1],
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            # third answer
            self.text_generation_3 = TextGeneration(
                name = "text_generation_04",
                llm = self.response_models[2],
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            self.combine_columns = CombineColumns(
                name="self.combine_columns",
                columns=["generation", "model_name"],
                output_columns=["generations", "model_name"],
                input_batch_size=BATCH_SIZE
            )

            self.keep_columns_1 = KeepColumns(
                columns = ["instruction", "generations"]
            )

            self.clean = CleanGenerations(
                name="clean_generations"
            )

            self.ultrafeedback = UltraFeedback(
                llm=self.ranking_model,
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                aspect="overall-rating",
                output_mappings={"model_name": "ultrafeedback_model"}
            )

            self.best_gen = SelectBestGeneration(
                name="select_best_gen"
            )

            self.keep_columns = KeepColumns(
                columns=["instruction", "generation"]
            )

            self.load_dataset >> [self.text_generation_1, self.text_generation_2, self.text_generation_3] >> self.combine_columns >> self.keep_columns_1 >> self.clean >> self.ultrafeedback >> self.best_gen >> self.keep_columns

    def process(self):
        distiset_1 = self.pipeline.run(
                parameters={
                    self.load_dataset.name: {
                        "repo_id": self.input_dataset,
                        "split": "train",
                    },
                    self.text_generation_1.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.text_generation_2.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.text_generation_3.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.ultrafeedback.name: {
                        "llm": {
                            "generation_kwargs": {
                                "max_new_tokens": self.config['max-tokens'],
                                "temperature": self.config['temperature'],
                            }
                        }
                    },
                },
            )

        distiset_1.push_to_hub(
            self.output_dataset,
            token = self.HF_AUTH_TOKEN
        )


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

    if max_competitor_rating > base_rating:
        best_competitor_index = competitor_ratings.index(max_competitor_rating) + 1
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

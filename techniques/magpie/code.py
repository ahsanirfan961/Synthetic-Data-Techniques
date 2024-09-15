from distilabel.llms import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import Magpie
from typing import List
from pydantic import Field
import yaml
from techniques.utilities import SplitList

class MagpieTechnique:
   
    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.magpie_model_config = next((model for model in config['models'] if model['type'] == 'magpie'), None)

        if self.magpie_model_config['vendor'] == 'openai':
            self.magpie_model = OpenAILLM(
               model=self.magpie_model_config['path'], 
               api_key=self.magpie_model_config['api_key'],
               generation_kwargs={
                    "temperature": config['temperature'],
                    "max_new_tokens": config['max_tokens'],
                }
            )
        elif self.magpie_model_config['vendor'] == 'huggingface':
            self.magpie_model = TransformersLLM(
                    model=self.magpie_model_config['path'],
                    magpie_pre_query_template=config['magpie_pre_query_template'],
                    generation_kwargs={
                        "temperature": config['temperature'],
                        "max_new_tokens": config['max_tokens'],
                    },
                    device="cuda:0",
                ),

        with Pipeline(name="Question Generation") as self.pipeline:
            self.load_hub_dataset = LoadDataFromHub(
                name="load_dataset",
            )

            self.magpie = Magpie(
                llm=self.magpie_model,
                # only_instruction=True,
                n_turns = config['conversation_turns']
            )

            self.instruct_res_pairs = ConversationToInstructPairs(
                name="conversation_to_instruct_pairs",
            )

            self.split_instructions = SplitList(
                name="split_instructions",
                split_column = 'instructions',
                output_mappings={"splitted": "instruction"}
            )

            self.split_responses = SplitList(
                name="split_responses",
                split_column = 'responses',
                output_mappings={"splitted": "response"}
            )

            self.keep_columns = KeepColumns(
                columns = ["instruction", "response"]
            )

            self.load_hub_dataset >> self.magpie >> self.instruct_res_pairs >> self.split_instructions >> self.split_responses >> self.keep_columns

    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_hub_dataset.name: {
                    "repo_id": self.input_dataset,
                    "split": "train",
                },
            },
        )

        distiset.push_to_hub(
           self.output_dataset,
           token=self.hf_token
        )


class ConversationToInstructPairs(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['conversation']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instructions', 'responses']

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
          instructions = []
          responses = []
          for talk in example['conversation']:
            if talk['role'] == 'user':
              instructions.append(talk['content'])
            else:
              responses.append(talk['content'])
          example['instructions'] = instructions
          example['responses'] = responses
          example.pop('conversation')
        yield inputs
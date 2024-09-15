import json
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from typing import List
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, PushToHub
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import TransformersLLM
from pydantic import BaseModel


class SentenceSimilarityTechnique:
   
    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.description_model = next((model['path'] for model in config['models'] if model['type'] == 'description'), None)

        with Pipeline(name='sentence-similarity-pipeline') as self.synthesis_pipeline:
            self.load_data_from_hub = LoadDataFromHub(
                name='load-data-from-hub',
            )

            generate_prompts = GeneratePrompts(
                name='generate-prompts',
                input_batch_size=self.config['input-batch-size']
            )

            self.synthesizer = TextGeneration(
                name='description-text-genration',
                llm = TransformersLLM(
                    model=self.description_model,
                    device="cuda:0",
                    structured_output={"format": "json", "schema": Description}
                ),
                output_mappings={"generation": "descriptions"},
                input_batch_size=self.config['input-batch-size']
            )

            parse_descriptions = ParseDescriptions(
                name='parse-descriptions',
                input_batch_size=self.config['input-batch-size']
            )

            keep_columns = KeepColumns(
               columns = ['text', 'good_description', 'bad_description']
            )

            push_to_hub = PushToHub(
               repo_id=self.output_dataset,
               token=self.hf_token
            )

            self.load_data_from_hub >> generate_prompts >> self.synthesizer >> parse_descriptions >> keep_columns >> push_to_hub

    def process(self):
        self.synthesis_pipeline.run(
            parameters={
                self.load_data_from_hub.name: {
                    'repo_id': self.input_dataset,
                    "split": "train"
                },
                self.synthesizer.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": self.config['max-tokens'],
                            "temperature": self.config['temperature'],
                        },
                    },
                }
            },
        )

class GeneratePrompts(Step):

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['text']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['text', 'instruction']

    def process(self, inputs: StepInput) -> StepOutput:

        for example in inputs:
          instruction = f"""
            Let's write abstract descriptions of sentences. Example:
            Sentence: Pilate's role in the events leading to the crucifixion lent themselves to melodrama , even tragedy , and Pilate often has a role in medieval mystery plays .
            Description: A description of a historical religious figure's involvement in a significant event and its later portrayal in art.
            Note: Descriptions can differ in the level of abstraction, granularity and the part of the sentence they focus on. Some descriptions need to be abstract, while others should be concrete and detailed.
            For the following sentence, write up 5 good and stand-alone, independent descriptions and 5 bad descriptions (which may be related, but are clearly wrong). Output a json file with keys 'good', 'bad'.
            Sentence: {example['text']}
            Start your answer with a curly bracket.
            """
          example['instruction'] = instruction

        yield inputs

class ParseDescriptions(Step):

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['descriptions']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['good_description', "bad_description"]

    def process(self, inputs: StepInput) -> StepOutput:

        new_generations = []
        for example in inputs:
          try:
            parsed = json.loads(example['descriptions'])
            for good, bad in zip(parsed['good'], parsed['bad']):
              new_generations.append({
                'text': example['text'],
                'instruction': example['instruction'],
                'good_description': good,
                'bad_description': bad
              })
          except:
            print("Can't parse descriptions")
            print(example['descriptions'])

        yield inputs

class Description(BaseModel):
    good: list[str]
    bad: list[str]


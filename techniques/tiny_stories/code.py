from pydantic import Field
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from typing import List
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, PushToHub
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import TransformersLLM, OpenAILLM
import pandas as pd
import random, itertools
from techniques.utilities import GeneratePrompts


class TinyStoriesTechnique:

    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.generation_model_config = next((model for model in config['models'] if model['type'] == 'generation'), None)

        if self.generation_model_config['vendor'] == 'openai':
            model = OpenAILLM(
                model=self.generation_model_config['path'], 
                api_key=self.generation_model_config['key'],
            )
        elif self.generation_model_config['vendor'] == 'huggingface':
            model = TransformersLLM(
                    model=self.generation_model_config['path'],
                    device="cuda:0",
            )
        
        with Pipeline(name='tiny-stories-pipeline') as self.tiny_pipeline:
            self.load_data_from_hub = LoadDataFromHub(
                name='load-data-from-hub',
            )

            select_random_values = SelectRandomValues(
                name='select-random-values',
                stories=5,
                input_batch_size = 4
            )

            generate_prompts = GeneratePrompts(
                name='generate-prompts',
                template="""
                    Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. The story should use the verb "{verb}", the noun "{noun}" and the adjective "{adjective}". Remember to only use simple words and only provide story text in response. No initial or ending explainatory texts are required!
                """
            )

            self.synthesizer = TextGeneration(
                name='story-genration',
                llm = TransformersLLM(
                    model="microsoft/Phi-3.5-mini-instruct",
                    device="cuda:0",
                ),
                input_batch_size = 4,
                output_mappings={"generation": "story"}
            )

            keep_columns = KeepColumns(
                name='keep-columns',
                columns=['noun', 'adjective', 'verb', 'story']
            )

            push_to_hub = PushToHub(
               repo_id=self.output_dataset,
               token=self.hf_token
            )

            self.load_data_from_hub >> select_random_values >> generate_prompts >> self.synthesizer >> keep_columns >> push_to_hub
    
    def process(self):
        self.tiny_pipeline.run(
            parameters={
                self.load_data_from_hub.name: {
                    'repo_id': "ahsanirfan961/noun-adj-verb",
                    "split": "train"
                },
                self.synthesizer.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": 256,
                            "temperature": 0.7,
                        },
                    },
                }
            },
        )


class SelectRandomValues(Step):

    stories: int = Field(..., description="The number of stories to select.")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['noun', 'adjective', 'verb']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['noun', 'adjective', 'verb']

    def process(self, inputs: StepInput) -> StepOutput:

        print(inputs)
        
        input_data = pd.DataFrame(inputs).to_dict(orient='list')
        combinations = list(itertools.product(input_data['noun'], input_data['adjective'], input_data['verb']))
        random.shuffle(combinations)
        selected_combinations = combinations[:self.stories]
        
        result = []
        for combination in selected_combinations:
            result.append({
                'noun': combination[0],
                'adjective': combination[1],
                'verb': combination[2]
            })
        yield result

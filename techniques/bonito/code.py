from bonito import Bonito
from vllm import SamplingParams
from datasets import load_dataset, Dataset
from pydantic import Field
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from typing import List
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts


class BonitoTechnique:
        
    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.instruct_model = next((model['path'] for model in config['models'] if model['type'] == 'instruct'), None)

        with Pipeline(name='bonito-pipeline') as self.pipeline:
            self.load_data_from_hub = LoadDataFromHub(
                name='load-data-from-hub',
                output_mappings={"text": "input"}
            )

            synthesizer = GenerateBonito(
                name='synthesizer',
                input_batch_size = config['input_batch_size'],
                max_tokens = config['max_tokens'],
                top_p = config['top_p'],
                temperature = config['temperature'],
                n = config['n_instructions'],
                context_col = 'input',
                task_type = config['task_type'],
                model = self.instruct_model,
                output_mappings={"input": "instruction", "output": "response"}
            )
            
            self.load_data_from_hub >> synthesizer 

    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_data_from_hub.name: {
                    'repo_id': self.input_dataset,
                    "split": "train"
                },
            },
        )

        distiset.push_to_hub(
            self.output_dataset, 
            token=self.hf_token
        )



class GenerateBonito(Step):

    max_tokens: int = Field(..., description="The maximum number of tokens per generation")
    top_p: float = Field(..., description="The top_p value for sampling")
    temperature: float = Field(..., description="The temperature value for sampling")
    n: int = Field(..., description="The number of synthetic examples to generate")
    context_col: str = Field(..., description="The column name of the context in the input data")
    task_type: str = Field(..., description="The type of task to generate")
    model: str = Field(..., description="The model id to use for generation")

    def __init__(self, name, **data):
        super().__init__(name=name, **data)
        self._bonito = None

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['input']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['input', 'output']

    def process(self, inputs: StepInput) -> StepOutput:
        
        if self._bonito is None:
            self._bonito = Bonito(self.model)

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens, 
            top_p=self.top_p, 
            temperature=self.temperature, 
            n=self.n
        )

        input_dataset = Dataset.from_list(inputs)
        
        synthetic_dataset = self._bonito.generate_tasks(
            input_dataset,
            context_col=self.context_col,
            task_type=self.task_type,
            sampling_params=sampling_params
        )
        
        yield synthetic_dataset.to_pandas().to_dict(orient='records')
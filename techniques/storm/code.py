from pydantic import Field
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
import yaml
from techniques.utilities import InstructClassification, FilterOnRanking

class StormTechnique:

    def __init__(self, config) -> None:
        
        self.config = config
        self.hf_token = config['hf-token']
        
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.classification_model = next((model['path'] for model in config['models'] if model['type'] == 'classification'), None)

        LABELS = config['labels']
        THRESHOLDS = config['thresholds']

        with Pipeline(name="Data Curation") as self.curation_pipeline:
            self.load_dataset = LoadDataFromHub(
                name="Load_Dataset",
                output_mappings={"instruction": "question", "generation": "answer"}
                )

            self.educative_classifier = InstructClassification(
                name="Educative_Classifier",
                candidate_labels=LABELS['educative'],
                input_batch_size=config['batch-size'],
                model=self.classification_model
            )

            self.educative_filter = FilterOnRanking(
                name="Educative_Filter",
                threshold=THRESHOLDS['educative'],
                output_mappings={"ranking": "educative"}
            )

            self.difficulty_classifier = InstructClassification(
                name="Difficulty_Classifier",
                candidate_labels=LABELS['difficulty'],
                input_batch_size=config['batch-size'],
                model=self.classification_model
            )

            self.difficulty_filter = FilterOnRanking(
                name="Difficulty_Filter",
                threshold=THRESHOLDS['difficulty'],
                output_mappings={"ranking": "difficulty"}
            )

            self.load_dataset >> self.educative_classifier >> self.educative_filter >> self.difficulty_classifier >> self.difficulty_filter

    def process(self):
        curated_data = self.curation_pipeline.run(
            parameters={
                self.load_dataset.name: {
                    "repo_id": self.input_dataset,
                    "split": "train",
                },
            },
        )

        curated_data.push_to_hub(
            self.output_dataset,
            token = self.HF_AUTH_TOKEN
        )

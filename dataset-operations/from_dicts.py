from datasets import Dataset

# Your data
data = [
    {"noun": "Mountain", "adjective": "Brave", "verb": "Run"},
    {"noun": "River", "adjective": "Quiet", "verb": "Jump"},
    {"noun": "Bicycle", "adjective": "Swift", "verb": "Create"},
    {"noun": "Elephant", "adjective": "Enormous", "verb": "Whisper"},
    {"noun": "Computer", "adjective": "Colorful", "verb": "Fly"},
    {"noun": "Book", "adjective": "Fragile", "verb": "Explore"},
    {"noun": "Ocean", "adjective": "Ancient", "verb": "Build"},
    {"noun": "Tree", "adjective": "Mysterious", "verb": "Capture"}
]

# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_list(data)

# Push to Hugging Face Hub
dataset.push_to_hub("ahsanirfan961/noun-adj-verb", token="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")

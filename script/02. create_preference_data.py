import concurrent.futures
import json
import re
from typing import List, Tuple
from datasets import Dataset
from openai import OpenAI
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

class PreferenceSet:
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PreferenceSet':
        data = json.loads(json_str)
        triples = [(triple['instruction'], triple['generated_answer'], triple['extracted_answer']) for triple in data['preference_triples']]
        return cls(triples)
    
    def __iter__(self):
        return iter(self.triples)
    

def load_articles_from_json(file_path: str) -> Dataset:
    with open(file_path, "r") as file:
        data = json.load(file)
    return Dataset.from_dict(
        {
        # "id": [item["id"] for item in data["artifact_data"]],
        # "content": [item["content"] for item in data["artifact_data"]],
        # "platform": [item["platform"] for item in data["artifact_data"]],
        # "author_id": [item["author_id"] for item in data["artifact_data"]],
        # "author_full_name": [item["author_full_name"] for item in data["artifact_data"]],
        # "link": [item["link"] for item in data["artifact_data"]],
        "id": [dat['_id'] for dat in data],
        "content": [dat['content'] for dat in data],
        "platform": [dat['platform'] for dat in data],
        "author_id": [dat['author_id'] for dat in data],
        "author_full_name": [dat['author_full_name'] for dat in data],
        "link": [dat['link'] for dat in data],

        }
    )

def clean_text(text):
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_substrings(dataset: Dataset, min_length: int = 1000, max_length: int = 2000) -> List[str]:
    extracts = []
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
    # print(dataset)
    for article in dataset["content"]:
        cleaned_article = clean_text(article['Content'])
        sentences = re.split(sentence_pattern, cleaned_article)
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
                current_chunk = sentence + " "
        if len(current_chunk) >= min_length:
            extracts.append(current_chunk.strip())
    return extracts


def generate_preference_triples(extract: str, client: OpenAI) -> List[Tuple[str, str, str]]:
    prompt = f"""
    Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
        1. An instruction asking about a specific topic in the context.
        2. A generated answer that attempts to answer the instruction based on the context.
        3. An extracted answer that is a relevant excerpt directly from the given context.
    
    Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.
    
    Important:
    - Ensure that the extracted answer is a verbatim copy from the context, including all punctuation and apostrophes.
    - Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
    - If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.
    
    Provide your response in JSON format with the following structure:
    {{
        "preference_triples": [
            {{
                "instruction": "...",
                "generated_answer": "...",
                "extracted_answer": "..."
            }},
            ...
        ]
    }}

    Extract:
    {extract}
    """
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who generates instruction-answer triples based on the given context. "
            "           Each triple should include an instruction, a generated answer, and an extracted answer from the context. Provide your response in JSON format.",
            
        },
            
        {
            "role": "user", 
            "content": prompt
         },
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
        temperature=0.7,
    )
    result = PreferenceSet.from_json(completion.choices[0].message.content)
    
    return result.triples

def filter_short_answers(dataset: Dataset, min_length: int = 100) -> Dataset:
    def is_long_enough(example):
        return len(example['chosen']) >= min_length
    
    return dataset.filter(is_long_enough)

def filter_answer_format(dataset: Dataset) -> Dataset:
    def is_valid_format(example):
        chosen = example['chosen']
        return (len(chosen) > 0 and
                chosen[0].isupper() and
            chosen[-1] in ('.', '!', '?'))
    
    return dataset.filter(is_valid_format)


def create_preference_dataset(dataset: Dataset, client: OpenAI, num_workers: int = 4) -> Dataset:
    extracts = extract_substrings(dataset)
    preference_triples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_preference_triples, extract, client)
            for extract in extracts
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
           preference_triples.extend(future.result())
           instructions, generated_answers, extracted_answers = zip(*preference_triples)
    
    return Dataset.from_dict(
        {
            "prompt": list(instructions),
            "rejected": list(generated_answers),
            "chosen": list(extracted_answers)
        }
    )


# def main(dataset_id: str) -> Dataset:
client = OpenAI()

# 1. Load the raw data
raw_dataset = load_articles_from_json("D:\Project\MirrorMuse\data\data_warehouse_raw_data\ArticleDocument.json")
print("Raw dataset:")
print(raw_dataset.to_pandas())
    

# 2. Create preference dataset
dataset = create_preference_dataset(raw_dataset, client)
print("Preference dataset:")
print(dataset.to_pandas())

# 3. Filter out samples with short answers
dataset = filter_short_answers(dataset)

# 4. Filter answers based on format
dataset = filter_answer_format(dataset)

# 5. Export
dataset.push_to_hub("SkillRipper/preference-data")
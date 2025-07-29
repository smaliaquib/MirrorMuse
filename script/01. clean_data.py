import concurrent.futures
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from datasets import Dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

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


class InstructionAnswerSet:
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> 'InstructionAnswerSet':
        data = json.loads(json_str)
        pairs = [(pair['instruction'], pair['answer'])
                 for pair in data['instruction_answer_pairs']]
        return cls(pairs)
    
    def __iter__(self):
        return iter(self.pairs)
    
def generate_instruction_answer_pairs(extract: str, client: OpenAI) -> List[Tuple[str, str]]:
    prompt = f"""Based on the following extract, generate five 
                instruction-answer pairs. Each instruction \
                must ask to write about a specific topic contained in the context. 
                each answer \
                must provide a relevant paragraph based on the information found in 
                the \
                context. Only use concepts from the context to generate the 
                instructions. \
                Instructions must never explicitly mention a context, a system, a 
                course, or an extract. \
                Instructions must be self-contained and general. \
                Answers must imitate the writing style of the context. \
                Example instruction: Explain the concept of an LLM Twin. \
                Example answer: An LLM Twin is essentially an AI character that 
                mimics your writing style, personality, and voice. \
                It's designed to write just like you by incorporating these elements 
                into a language model. \
                The idea is to create a digital replica of your writing habits using 
                advanced AI techniques. \
                Provide your response in JSON format with the following structure:
                {{
                    "instruction_answer_pairs": [
                        {{"instruction": "...", "answer": "..."}},
                        ...
                    ]
                }}
                Extract:
                {extract}
                """
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
                {"role": "system", "content": "You are a helpful assistant who generates instruction-answer pairs based on the given context. Provide your response in JSON format."},
                {"role": "user", "content": prompt},
            ],
        response_format={"type": "json_object"},
        max_tokens=1200,
        temperature=0.7,
    )
    
    # Parse the structured output
    result = InstructionAnswerSet.from_json(completion.choices[0].message.content)

    # Convert to list of tuples
    return result.pairs

def create_instruction_dataset(dataset: Dataset, client: OpenAI, num_workers: int = 4) -> Dataset:
    extracts = extract_substrings(dataset)
    instruction_answer_pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_instruction_answer_pairs, extract, client)
                   for extract in extracts
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            instruction_answer_pairs.extend(future.result())
    instructions, answers = zip(*instruction_answer_pairs)
    return Dataset.from_dict({"instruction": list(instructions), "output": list(answers)})


client = OpenAI()
    
# 1. Load the raw data
raw_dataset = load_articles_from_json("D:\prod\MirrorMuse\data\data_warehouse_raw_data\ArticleDocument.json")
print("Raw dataset:")

print(raw_dataset.to_pandas())

# 2. Create instructiondataset
instruction_dataset = create_instruction_dataset(raw_dataset, client)
print("Instruction dataset:")
print(instruction_dataset.to_pandas())

filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)
filtered_dataset.push_to_hub("SkillRipper/llmtwin")


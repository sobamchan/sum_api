import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (AutoTokenizer, BartConfig,
                          BartForConditionalGeneration)

app = FastAPI()


class Text(BaseModel):
    document: str


class Model:
    def __init__(self):
        print("initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        bart = BartForConditionalGeneration(BartConfig())
        bart.load_state_dict(torch.load("./model/model.state"), strict=False)
        self.bart = bart
        print("loaded!")

    def summarize(self, text: Text):

        inputs = self.tokenizer(
            [text.document], padding="max_length", truncation=True, return_tensors="pt"
        )
        summary_ids = self.bart.generate(
            inputs["input_ids"],
            max_length=50,
            num_beams=1,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]


model = Model()


@app.post("/summarize")
def summarize(text: Text):
    return model.summarize(text)

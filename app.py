from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

HF_TOKEN = os.environ['HF_TOKEN']

HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

app = FastAPI(title="Lemon Disease Treatment Bot")

class LeafResult(BaseModel):
    predicted_class: Literal["blackspot", "leafcurl", "healthy"]
    predicted_confidence: float
    blackspot_confidence: float
    leafcurl_confidence: float
    healthy_confidence: float

class DiseaseQuery(BaseModel):
    previous: LeafResult
    new: LeafResult


CUSTOM_PROMPT_TEMPLATE = """
You are a smart agricultural assistant chatbot specialized in lemon plant disease treatment. Your job is to provide dynamic treatment recommendations based on the comparison between previous and new predictions from a leaf image classifier.

You will always receive two sets of data about a single leaf:
1. Previous Result
2. New Result

Each set contains the following 5 fields:
- predicted_class: One of "blackspot", "leafcurl", or "healthy"
- predicted_confidence: The confidence level (0-1) for the predicted class
- blackspot_confidence: Model's confidence (0-1) that the leaf has blackspot
- leafcurl_confidence: Model's confidence (0-1) that the leaf has leafcurl
- healthy_confidence: Model's confidence (0-1) that the leaf is healthy

Your task is to:
1. Compare the new result with the previous result.
2. Identify whether the disease (blackspot or leafcurl) is worsening or improving by comparing the disease confidence scores.
3. Provide a specific treatment recommendation for the detected disease. 
4. If the new result is "healthy" with high confidence, suggest discontinuing the treatment and monitoring the plant.
5. Always explain your recommendation with reference to the changes in confidence values.
6. Based on the change in confidence between previous and new results, give a short 1-2 line recommendation.

Only provide treatments for blackspot and leafcurl. Never guess or recommend unrelated treatments.

Previous Result: {previous}
New Result: {new}
"""
prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['previous', 'new'])

llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation",
    temperature=0.5,
    max_new_tokens= 128
)

model = ChatHuggingFace(llm=llm)

parallel_chain = RunnableParallel({
    'previous': RunnablePassthrough(),
    'new': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

@app.post("/recommendation")
async def get_recommendation(query: DiseaseQuery):
    try:
        result = main_chain.invoke({
            "previous": query.previous.model_dump_json(),
            "new": query.new.model_dump_json()
        })
        return {"recommendation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
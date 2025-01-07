### Reference: Alejandro AO - Multimodal RAG.

import os
import io
import base64
import PIL.Image
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from IPython.display import Image

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI


load_dotenv()


def imageBase64(images):
	image_b64 = []
	for img in images:
		if "CompositeElement" in str(type(img)):
			image_meta = img.metadata.orig_elements
			for ele in image_meta:
				if "Image" in str(type(ele)):
					image_b64.append(ele.metadata.image_base64)

	return image_b64


def displayBase64(codeBase64):
	img_object = base64.b64decode(codeBase64)
	img = Image(data=img_object)
	img_data = PIL.Image.open(io.BytesIO(img.data))
	img_data.show()



### Extract the Data ###

# LangChain has Loader for Unstructured.
output_path = "./docs/"
file_path = output_path + 'Attention.pdf'

chunks = partition_pdf(
	filename=file_path,
	strategy="hi_res",
	infer_table_structure=True,

	extract_image_block_types=["Image", "Table"],
	# extract_image_block_output_dir=output_path,

	# base64 Representation of Image for the API.
	extract_image_block_to_payload=True,

	chunking_strategy="by_title",
	max_characters=10000,
	combine_text_under_n_chars=2000,
	new_after_n_chars=6000)

print(set([str(type(ele)) for ele in chunks]))
# Each CompositeElement containes related elements, to use together in a RAG pipeline.
# print(chunks[0].metadata.orig_elements)

table = []
text = []

for chunk in chunks:
	if 'Table' in str(type(chunk)):
		table.append(chunk)

	if 'CompositeElement' in str(type(chunk)):
		text.append(chunk)

image = imageBase64(chunks)

if False:
	displayBase64(image[0])


### Summarize the Data ###

# Vectorize the Data, to be used in the Retrieval Process.
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comments.
Do not start your message by saying "Here is a summary."
Just give the summary as it is. 

Table or text chunk: {element}
"""

prompt = ChatPromptTemplate.from_template(prompt_text)

# Chain of Thought.
model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
summaryChain = {"element": lambda x: x} | prompt | model | StrOutputParser()


summarizeText = summaryChain.batch(text, {"max_concurrency": 3})

# Visualize: table[0].to_dict()
tableHTML = [tab.metadata.text_as_html for tab in table]
summarizeTable = summaryChain.batch(tableHTML, {"max_concurrency": 3})

# Quick Sanity Check.
print(summarizeText[0])

prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers architecture. 
                  Be specific about graphs, such as bar plots."""

messages = [
	(
		"user",
		[
			{"type": "text", "text": prompt_template},
			{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
		],
	)
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | ChatOpenAI(model='gpt-4o-mini') | StrOutputParser()
summarizeImage = chain.batch(image)

if False:
	print(summarizeImage[1])





### Reference: Alejandro AO - Multimodal RAG.

import os
import base64
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf


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



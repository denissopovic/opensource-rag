# AI-Talk Project

This project is designed to test and demonstrate the capabilities of language models using the `llama_index` module.

## Project Overview

The primary goal of this project is to run simple scripts to build a simple RAG-application. 

The `1_test_ollama.py` script is designed to perform a basic check to confirm that the LLM is installed with Ollama.

The `2_load_data.py` script is embedding the PDF-data into Qdrant vector db.

The `3_verify_index.py` script is to verify that the data we vectorized in Qdrant works.

The `app.py` runs a Flask server that runs an API so you can use Curl for example to test your RAG app.

  `curl --request POST 'http://127.0.0.1:5000/process_form' --form 'query=Who is Denis?'`

## Links

Learn more about prompt engineering
https://www.promptingguide.ai/

LlamaIndex
https://www.llamaindex.ai/

Malmö AI devs Meetup group
https://www.meetup.com/malmo-ai-devs/

Malmö AI devs Discord
https://discord.gg/8XbJKhXr

My LinkedIn
https://www.linkedin.com/in/denis-sopovic/
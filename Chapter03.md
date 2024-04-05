# Exploring API model integrations
For LLMs, OpenAI, Hugging Face, Cohere, Anthropic, Azure, Google Cloud Platform’s Vertex AI (PaLM-2), and Jina AI are among the many providers supported in LangChain; however, this list is growing all the time. You can check out the full list of supported integrations for LLMs at https://integrations.langchain.com/llms.

LangChain implements three different interfaces – we can use [chat models](https://python.langchain.com/docs/integrations/chat/), [LLMs](https://python.langchain.com/docs/integrations/llms/), and [embedding models](https://python.langchain.com/docs/integrations/text_embedding/).

Chat models and LLMs are similar in that they both process text input and produce text output. However, there are some differences in the types of input and output they handle. Chat models are specifically designed to handle a list of chat messages as input and generate a chat message as output. They are commonly used in chatbot applications where conversations are exchanged. You can find chat models at https://python.langchain.com/docs/integrations/chat.

Finally, text embedding models are used to convert text inputs into numerical representations called **embeddings**. We’ll focus on text generation in this chapter, and discuss embeddings, vector databases, and neural search in Chapter 5, Building a Chatbot Like ChatGPT. Suffice it to say here that these embeddings are a way to capture and extract information from the input text. They are widely used in natural language processing tasks like sentiment analysis, text classification, and information retrieval. Embedding models are listed at https://python.langchain.com/docs/integrations/text_embedding.

## OpenAPI Key
For each of these providers, to make calls against their API, you’ll first need to create an account and obtain an API key. This is free of charge for all providers and, with some of them, you don’t even have to give them your credit card details.

To set an API key in an environment, in Python, we can execute the following lines:
```python
import os
os.environ["OPENAI_API_KEY"] = "<your token>"
```
Here, `OPENAI_API_KEY` is the environment key that is appropriate for OpenAI. Setting the keys in your environment has the advantage of not needing to include them as parameters in your code every time you use a model or service integration.

You can also expose these variables in your system environment from your terminal. In Linux and macOS, you can set a system environment variable from the terminal using the `export` command:
```bash
export OPENAI_API_KEY=<your token>
```
To permanently set the environment variable in Linux or macOS, you would need to add the preceding line to the `~/.bashrc` or `~/.bash_profile` file, respectively, and then reload the shell using the command source `~/.bashrc` or `source ~/.bash_profile`.

In Windows, you can set a system environment variable from the command prompt using the `set` command:
```bash
set OPENAI_API_KEY=<your token>
```

To permanently set the environment variable in Windows, you can add the preceding line to a batch script.

My personal choice is to create a `config.py` file, where all the keys are stored. I then import a function from this module that will load all these keys into the environment. If you look for this file in the Github repository, you’ll notice that it is missing. This is on purpose (in fact, I’ve disabled the tracking of this file in Git) since I don’t want to share my keys with other people for security reasons (and because I don’t want to pay for anyone else’s usage).

My `config.py` looks like this:
```python
import os
OPENAI_API_KEY = "... "
# I'm omitting all other keys
def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
```

You can set all your keys in the `config.py` file. This function, `set_environment()`, loads all the keys into the environment as mentioned. Anytime you want to run an application, you import the function and run it like so:
```python
from config import set_environment
set_environment()
```
Now, let’s go through a few prominent model providers in turn. We’ll give an example of usage for each of them. Let’s start with a fake LLM that we can use for testing purposes. This will help to illustrate the general idea of calling language models in LangChain.
## Fake LLM
The fake LLM allows you to simulate LLM responses during testing without needing actual API calls. This is useful for rapid prototyping and unit testing agents. Using the FakeLLM avoids hitting rate limits during testing. It also allows you to mock various responses to validate that your agent handles them properly. Overall, it enables fast agent iteration without needing a real LLM.

For example, you could initialize a FakeLLM that returns `"Hello"` as follows:
```python
from langchain.llms import FakeLLM
fake_llm = FakeLLM(responses=["Hello"])
```

You can execute this example in either Python directly or in a notebook.

The fake LLM is only for testing purposes. The LangChain documentation has an example of tool use with LLMs. This is a bit more complex than the previous example but gives a hint of the capabilities we have at our fingertips:
```python
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
tools = load_tools(["python_repl"])
responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("whats 2 + 2")
```

We set up an agent that makes decisions based on the React strategy that we explained in Chapter 2, LangChain for LLM Apps (`ZERO_SHOT_REACT_DESCRIPTION`). We run the agent with a text: the question `what's 2 + 2`.

As you can see, we connect a tool, a **Python Read-Eval-Print Loop (REPL)**, that will be called depending on the output of the LLM. `FakeListLLM` will give two responses (`"Action: Python_REPL\nAction Input: print(2 + 2)"` and `"Final Answer: 4"`) that won’t change based on the input.

We can also observe how the fake LLM output leads to a call to the Python interpreter, which returns 4. Please note that the action must match the name attribute of the tool, PythonREPLTool, which starts like this:

class PythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""
    name = "Python_REPL"
    description = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
As you can see in the preceding code block, the names and descriptions of the tools are passed to the LLM, which then decides an action based on the information provided. The action can be executing a tool or planning.

The output of the Python interpreter is passed to the fake LLM, which ignores the observation and returns `4`. Obviously, if we change the second response to `"Final Answer: 5"`, the output of the agent wouldn’t correspond to the question.

## OpenAI
OpenAI also offers an **Embedding** class for text embedding models.

We can use the `OpenAI` language model class to set up an LLM to interact with. Let’s create an agent that calculates using this model – I am omitting the imports from the previous example:
```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0., model="text-davinci-003")
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("whats 4 + 4")
```
We should be seeing this output:
```bash
> Entering new  chain...
 I need to add two numbers
Action: Python_REPL
Action Input: print(4 + 4)
Observation: 8
Thought: I now know the final answer
Final Answer: 4 + 4 = 8
> Finished chain.
'4 + 4 = 8'
```
## Hugging Face
Hugging Face is a very prominent player in the NLP space and has considerable traction in open-source and hosting solutions.

These tools allow users to load and use models, embeddings, and datasets from Hugging Face. The `HuggingFaceHub` integration, for example, provides access to different models for tasks like text generation and text classification. The `HuggingFaceEmbeddings` integration allows users to work with sentence-transformer models.

Hugging Face offer various other libraries within their ecosystem, including `Datasets` for dataset processing, `Evaluate` for model evaluation, `Simulate` for simulation, and `Gradio` for machine learning demos.

To use Hugging Face as a provider for your models, you can create an account and API keys at https://huggingface.co/settings/profile. Additionally, you can make the token available in your environment as `HUGGINGFACEHUB_API_TOKEN`.

Let’s see an example, where we use an open-source model developed by Google, the Flan-T5-XXL model:
```python
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(
    model_kwargs={"temperature": 0.5, "max_length": 64},
    repo_id="google/flan-t5-xxl"
)
prompt = "In which country is Tokyo?"
completion = llm(prompt)
print(completion)
```
We get the response `"japan"`.
## Jina AI
You can set up a login at https://chat.jina.ai/api.

On the platform, we can set up APIs for different use cases such as image caption, text embedding, image embedding, visual question answering, visual reasoning, image upscale, or Chinese text embedding.

We get examples for client calls in Python and cURL, and a demo, where we can ask a question. This is cool, but unfortunately, these APIs are not available yet through LangChain. We can implement such calls ourselves by subclassing the `LLM` class in LangChain as a custom LLM interface.

Let’s set up another chatbot, this time powered by Jina AI. We can generate the API token, which we can set as `JINACHAT_API_KEY`, at https://chat.jina.ai/api.

Let’s translate from English to French here:
```python
from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage
chat = JinaChat(temperature=0.)
messages = [
    HumanMessage(
        content="Translate this sentence from English to French: I love generative AI!"
    )
]
chat(messages)
```
We should be seeing :
```bash
AIMessage(content="J'adore l'IA générative !", additional_kwargs={}, example=False).
```

We can set different temperatures, where a low temperature makes the responses more predictable. In this case, it makes only a minor difference. We are starting the conversation with a system message clarifying the purpose of the chatbot.

Let’s ask for some food recommendations:
```python
from langchain.schema import SystemMessage
chat = JinaChat(temperature=0.)
chat(
    [
        SystemMessage(
            content="You help a user find a nutritious and tasty food to eat in one word."
        ),
        HumanMessage(
            content="I like pasta with cheese, but I need to eat more vegetables, what should I eat?"
        )
    ]
)
```

I get this response in Jupyter – your answer could vary:
```bash
AIMessage(content='A tasty and nutritious option could be a vegetable pasta dish. Depending on your taste, you can choose a sauce that complements the vegetables. Try adding broccoli, spinach, bell peppers, and zucchini to your pasta with some grated parmesan cheese on top. This way, you get to enjoy your pasta with cheese while incorporating some veggies into your meal.', additional_kwargs={}, example=False)
```

It ignored the one-word instruction, but I liked reading the ideas. I think I should try this for my son. With other chatbots, I got `Ratatouille` as a suggestion.

It’s important to understand the difference in LangChain between LLMs and chat models. LLMs are text completion models that take a string prompt as input and output a string completion. As mentioned, chat models are like LLMs but are specifically designed for conversations. They take a list of chat messages as input, labeled with the speaker, and return a chat message as output.

Both LLMs and chat models implement the base language model interface, which includes methods such as `predict()` and `predict_messages()`. This shared interface allows for interchangeability between diverse types of models in applications and between chat and LLM models.
## Replicate
Replicate has lots of models available on their platform: https://replicate.com/explore.
You can authenticate with your GitHub credentials at https://replicate.com/. If you then click on your user icon at the top left, you’ll find the API tokens – just copy the API key and make it available in your environment as `REPLICATE_API_TOKEN`. To run bigger jobs, you need to set up your credit card (under *billing*).

Here is a simple example for creating an image:
```python
from langchain.llms import Replicate
text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    input={"image_dimensions": "512x512"},
)
image_url = text2image("a book cover for a book about creating generative ai applications in Python")
```

# Exploring local models
> Please note that we don’t need an API token for local models!

Let’s preface this with a note of caution: an LLM is big, which means that it’ll take up a lot of disk space or system memory. The use cases presented in this section should run even on old hardware, like an old MacBook; however, if you choose a big model, it can take an exceptionally long time to run or may crash the Jupyter notebook. One of the main bottlenecks is memory requirement. 

You can also run these models on hosted resources or services such as Kubernetes or Google Colab. These will let you run on machines with a lot of memory and different hardware including **Tensor Processing Units (TPUs)** or GPUs.

We’ll have a look here at Hugging Face’s `transformers`, `llama.cpp`, and GPT4All. These tools provide huge power and are full of great functionality too broad to cover in this chapter. Let’s start by showing how we can run a model with the transformers library by Hugging Face.

## Hugging Face Transformers
I’ll quickly show the general recipe for setting up and running a pipeline:
```python
from transformers import pipeline
import torch
generate_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
)
generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")
```
Running the preceding code will download everything that’s needed for the model such as the tokenizer and model weights from Hugging Face. This model is quite small (355 million parameters) but relatively performant and instruction-tuned for conversations. We can then run a text completion to give us some inspiration for this chapter.
> I haven’t included `accelerate` in the main requirements, but I’ve included the transformers library. If you don’t have all libraries installed, make sure you execute this command:
> ```bash
> pip install transformers accelerate torch
> ```
To plug this pipeline into a LangChain agent or chain, we can use it the same way that we’ve seen in the other examples in this chapter:
```python
from langchain import PromptTemplate, LLMChain
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=generate_text)
question = "What is electroencephalography?"
print(llm_chain.run(question))
```
In this example, we also see the use of a `PromptTemplate` that gives specific instructions for the task.

`llama.cpp` is a C++ port of Facebook’s LLaMA, LLaMA 2, and other derivative models with a similar architecture. Let’s have a look at this next.
## llama.cpp
Written and maintained by Georgi Gerganov, `llama.cpp` is a C++ toolkit that executes models based on architectures based on or like LLaMA, one of the first large open-source models, which was released by Meta, and which spawned the development of many other models in turn. One of the main use cases of `llama.cpp` is to run models efficiently on the CPU; however, there are also some options for GPU.

Please note that you need to have an **md5 checksum** tool installed. This is included by default in several Linux distributions such as Ubuntu. On macOS, you can install it with `brew` like this:
```bash
brew install md5sha1sum
```
We need to download the llama.cpp repository from GitHub. You can do this online by choosing one of the download options on GitHub, or you can use a `git` command from the terminal like this:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
```
Then we need to install the Python requirements, which we can do with the pip package installer – let’s also switch to the `llama.cpp` project root directory for convenience:
```bash
cd llama.cpp
pip install -r requirements.txt
```
You might want to create a Python environment before you install the requirements, but this is up to you. In my case, I received an error message at the end that a few libraries were missing, so I had to execute this command:
```bash
pip install 'blosc2==2.0.0' cython FuzzyTM
```
Now we need to compile `llama.cpp`. We can parallelize the build with `4` processes:
```bash
make -C . -j4 # runs make in subdir with 4 processes
```
To get the Llama model weights, you need to sign up with the T&Cs and wait for a registration email from Meta. There are tools such as the `llama` model downloader in the `pyllama` project, but please be advised that they might not conform to the license stipulations by Meta.

There are also many other models with more permissive licensing such as Falcon or Mistral, Vicuna, OpenLLaMA, or Alpaca. Let’s assume you download the model weights and the tokenizer model for the OpenLLaMA 3B model using the link on the llama.cpp GitHub page. The model file should be about 6.8 Gigabyes big, the tokenizer is much smaller. You can move the two files into the `models/3B` directory.

You can download models in much bigger sizes such as 13B, 30B, and 65B; however, a note of caution is in order here: these models are big both in terms of memory and disk space. We have to convert the model to llama.cpp format, which is called `ggml`, using the `convert` script:
```bash
python3 convert.py models/3B/ --ctx 2048. 
```
Then we can optionally quantize the models to save memory when doing inference. Quantization refers to reducing the number of bits that are used to store weight:
```bash
./quantize ./models/3B/ggml-model-f16.gguf ./models/3B/ggml-model-q4_0.bin q4_0
```
This last file is much smaller than the previous files and will take up much less space in memory as well, which means that you can run it on smaller machines. Once we have chosen a model that we want to run, we can integrate it into an agent or a chain, for example, as follows:
```bash
llm = LlamaCpp(
    model_path="./ggml-model-q4_0.bin",
    verbose=True
)
```
GPT4All Is a fantastic tool that not only includes running but also serving and customizing models.
## GPT4All
This tool is closely related to llama.cpp, and it’s based on an interface with llama.cpp. Compared to llama.cpp, however, it’s much more convenient to use and much easier to install. The setup instructions for this book already include the `gpt4all` library, which is needed.

As for model support, GPT4All supports a large array of Transformer architectures:
* GPT-J
* LLaMA (via llama.cpp)
* Mosaic ML’s MPT architecture
* Replit
* Falcon
* BigCode’s StarCoder

You can find a list of all available models on the project website, where you can also see their results in important benchmarks: https://gpt4all.io/.

Here’s a quick example of text generation with GPT4All:
```python
from langchain.llms import GPT4All
model = GPT4All(model="mistral-7b-openorca.Q4_0.gguf", n_ctx=512, n_threads=8)
response = model(
    "We can run large language models locally for all kinds of applications, "
)
```
Executing this should first download (if not downloaded yet) the model, which is one of the best chat model available through GPT4All, pre-trained by the French startup Mistral AI, and fine-tuned by the OpenOrca AI initiative. This model requires 3.83 GB of harddisk to store and 8 GB of RAM to run. Then we should hopefully see some convincing arguments for running LLMs locally.

This should serve as a first introduction to integrations with local models. In the next section, we’ll discuss building a text classification application in LangChain to assist customer service agents. The goal is to categorize customer emails based on intent, extract sentiment, and generate summaries to help agents understand and respond faster.

# Building an application for customer service
Customer service agents are responsible for answering customer inquiries, resolving issues, and addressing complaints. Their work is crucial for maintaining customer satisfaction and loyalty, which directly affects a company’s reputation and financial success.

Generative AI can assist customer service agents in several ways:
* **Sentiment classification**: This helps identify customer emotions and allows agents to personalize their responses.
* **Summarization**: This enables agents to understand the key points of lengthy customer messages and save time.
* **Intent classification**: Similar to summarization, this helps predict the customer’s purpose and allows for faster problem-solving.
* **Answer suggestions**: This provides agents with suggested responses to common inquiries, ensuring that accurate and consistent messaging is provided.

These approaches combined can help customer service agents respond more accurately and in a timely manner, improving customer satisfaction. Customer service is crucial for maintaining customer satisfaction and loyalty. Generative AI can help agents in several ways – sentiment analysis to gauge emotion, summarization to identify key points, and intent classification to determine purpose. Combined, these can enable more accurate, timely responses.

LangChain provides the flexibility to leverage different models. LangChain comes with many integrations that can enable us to tackle a wide range of text problems. We have a choice between many different integrations to perform these tasks.

We can access all kinds of models for open-domain classification and sentiment and smaller transformer models through Hugging Face for focused tasks. We’ll build a prototype that uses sentiment analysis to classify email sentiment, summarization to condense lengthy text, and intent classification to categorize the issue.

Given a document such as an email, we want to classify it into different categories related to intent, extract the sentiment, and provide a summary. We will work on other projects for question-answering in *Chapter 5, Building a Chatbot Like ChatGPT*.

We could ask any LLM to give us an open-domain (any category) classification or choose between multiple categories. In particular, because of their large training size, LLMs are enormously powerful models, especially when given few-shot prompts, for sentiment analysis that don’t need any additional training. This was analyzed by Zengzhi Wang and others in their April 2023 study, *Is ChatGPT a Good Sentiment Analyzer? A Preliminary Stud*y.

A prompt for an LLM for sentiment analysis could be something like this:
```bash
Given this text, what is the sentiment conveyed? Is it positive, neutral, or negative?
Text: {sentence}
Sentiment:
```
LLMs can also be highly effective at summarization, much better than any previous models. The downside can be that these model calls are slower than more traditional machine learning models and more expensive.

If we want to try out more traditional or smaller models, we can rely on libraries such as spaCy or access them through specialized providers. Cohere and other providers have text classification and sentiment analysis as part of their capabilities. For example, NLP Cloud’s model list includes spaCy and many others: https://docs.nlpcloud.com/#models-list.

Many Hugging Face models are supported for these tasks, including:
* Document question-answering
* Summarization
* Text classification
* Text question-answering
* Translation

We can execute these models either locally by running a `pipeline` in transformer, remotely on the Hugging Face Hub server (`HuggingFaceHub`), or as a tool through the `load_huggingface_tool()` loader.

Hugging Face contains thousands of models, many fine-tuned for particular domains. For example, **ProsusAI/finbert** is a BERT model that was trained on a dataset called **Financial PhraseBank** and can analyze the sentiment of financial text. We could also use any local model. For text classification, the models tend to be much smaller, so this would be less of a drag on resources. Finally, text classification could also be a case for embeddings, which we’ll discuss in *Chapter 5, Building a Chatbot Like ChatGPT*.

I’ve decided to try and manage as much as I can with smaller models that I can find on Hugging Face for this exercise.

We can list the 5 most downloaded models on Hugging Face Hub for text classification through the Hugging Face API:
```python
from huggingface_hub import list_models
def list_most_popular(task: str):
    for rank, model in enumerate(
        list_models(filter=task, sort="downloads", direction=-1)
):
        if rank == 5:
            break
        print(f"{model.id}, {model.downloads}\n")
list_most_popular("text-classification")
```
Let’s see the list:
| Model                                           | Downloads |
|-------------------------------------------------|-----------|
| distilbert-base-uncased-finetuned-sst-2-english | 40672289  |
| cardiffnlp/twitter-roberta-base-sentiment       | 9292338   |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli    | 7907049   |
| cardiffnlp/twitter-roberta-base-irony           | 7023579   |
| SamLowe/roberta-base-go_emotions                | 6706653   |

Generally, we should see that these models are about small ranges of categories such as sentiment, emotions, irony, or well-formedness. Let’s use a sentiment model with a customer email, which should be a common use case in customer service.

I’ve asked GPT-3.5 to put together a rambling customer email complaining about a coffee machine – I’ve shortened it a bit here. You can find the full email on GitHub. Let’s see what our sentiment model has to say:
```python
from transformers import pipeline
customer_email = """
I am writing to pour my heart out about the recent unfortunate experience I had with one of your coffee machines that arrived broken. I anxiously unwrapped the box containing my highly anticipated coffee machine. However, what I discovered within broke not only my spirit but also any semblance of confidence I had placed in your brand.
Its once elegant exterior was marred by the scars of travel, resembling a war-torn soldier who had fought valiantly on the fields of some espresso battlefield. This heartbreaking display of negligence shattered my dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable
"""
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)
print(sentiment_model(customer_email))
```

The sentiment model we are using here, Twitter-roBERTa-base, was trained on tweets, so it might not be the most adequate use case. Apart from emotion sentiment analysis, this model can also perform other tasks such as emotion recognition (anger, joy, sadness, or optimism), emoji prediction, irony detection, hate speech detection, offensive language identification, and stance detection (favor, neutral, or against).

For the sentiment analysis, we’ll get a rating and a numeric score that expresses confidence in the label. These are the labels:
* 0 – negative
* 1 – neutral
* 2 – positive

Please make sure you have all the dependencies installed according to instructions in order to execute this. I am getting this result:
```python
[{'label': 'LABEL_0', 'score': 0.5822020173072815}]
```
Not a happy camper.

For comparison, if the email says “I am so angry and sad, I want to kill myself,” we should get a score of close to 0.98 for the same label. We could try out other models or train better models once we have established metrics to work against.

Let’s move on!

Here are the 5 most popular models for summarization as well (downloads at the time of writing, October 2023):
| Model                         | Downloads |
|-------------------------------|-----------|
| facebook/bart-large-cnn       | 4637417   |
| t5-small                      | 2492451   |
| t5-base                       | 1887661   |
| sshleifer/distilbart-cnn-12-6 | 715809    |
| t5-large                      | 332854    |

All these models have a small footprint, which is nice, but to apply them in earnest, we should make sure they are reliable enough.

Let’s execute the summarization model remotely on a server. Please note that you need to have your `HUGGINGFACEHUB_API_TOKEN` set for this to work:
```python
from langchain import HuggingFaceHub
summarizer = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature":0, "max_length":180}
)
def summarize(llm, text) -> str:
    return llm(f"Summarize this: {text}!")
summarize(summarizer, customer_email)
```
After executing this, I see this summary:
```bash
A customer's coffee machine arrived ominously broken, evoking a profound sense of disbelief and despair. "This heartbreaking display of negligence shattered my dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable," the customer writes. "I hope this email finds you amidst an aura of understanding, despite the tangled mess of emotions swirling within me as I write to you," he adds.
```
This summary is just passable, but not very convincing. There is still a lot of rambling in the summary. We could try other models or just go for an LLM with a prompt asking to summarize. We’ll look at summarization in much more detail in *Chapter 4, Building Capable Assistants*. Let’s move on.

It could be quite useful to know what kind of issue the customer is writing about. Let’s ask Vertex AI:
> Before you execute the following code, make sure you have authenticated with GCP and you’ve set your GCP project according to the instructions mentioned in the section about Vertex AI.
> ```python
> from langchain.llms import VertexAI
> from langchain import PromptTemplate, LLMChain
> template = """Given this text, decide what is the issue the customer is concerned about. Valid > categories are these:
> * product issues
> * delivery problems
> * missing or late orders
> * wrong product
> * cancellation request
> * refund or exchange
> * bad support experience
> * no clear reason to be upset
> Text: {email}
> Category:
> """
> prompt = PromptTemplate(template=template, input_variables=["email"])
> llm = VertexAI()
> llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
> print(llm_chain.run(customer_email))
> ```

We get `product issues` back, which is correct for the long email example that I am using here.

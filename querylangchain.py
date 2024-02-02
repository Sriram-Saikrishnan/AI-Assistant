from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q2_K.bin', callbacks=[StreamingStdOutCallbackHandler()])

def answer_llm(text_qn):
    template = """
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Your answers are always short not more than 2 lines.
    <</SYS>>
    {text}[/INST]
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(text_qn)
    print(response)
    return response

#qn = "what is a large language model?"
#answer_llm(qn)
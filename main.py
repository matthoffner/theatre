from llama_index import (
    GPTSimpleVectorIndex, GPTListIndex, Document, ServiceContext, PromptHelper
)
from llama_index.indices.base import BaseGPTIndex
from llama_index.data_structs import Node
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index import LangchainEmbedding
from llama_index import LLMPredictor, ServiceContext, GPTListIndex
from langchain.llms import LlamaCpp
from adapter import HuggingFaceEmbeddings
from collections import deque
from pydantic import BaseModel, Field
from typing import Optional, Dict
import argparse


def format_text(text: str, user: str) -> str:
    return user + ": " + text


DEFAULT_USER_PREFIX_TMPL = (
    "Your name is {name}. "
    "We provide conversation context between you and other users below. "\
    "You are at work with someone else. \n"
    # "The user is the plaintiff and the other user is the defendant."
)
DEFAULT_PROMPT_TMPL = (
    "{context_str}"
    "Given the context information, perform the following task.\n"
    "Task: {query_str}\n"
    "You: "
    # "Here's an example:\n"
    # "Previous line: Hi Bob, good to meet you!\n"
    # "You: Good to meet you too!\n\n"
    # "Previous line: {query_str}\n"
    # "You: "
)
DEFAULT_PROMPT = QuestionAnswerPrompt(DEFAULT_PROMPT_TMPL)

class ConvoAgent(BaseModel):
    """Basic abstraction for a conversation agent."""
    name: str
    st_memory: deque
    lt_memory: BaseGPTIndex
    lt_memory_query_kwargs: Dict = Field(default_factory=dict)
    service_context: ServiceContext
    st_memory_size: int = 20
    # qa_prompt: QuestionAnswerPrompt = DEFAULT_PROMPT
    user_prefix_tmpl: str = DEFAULT_USER_PREFIX_TMPL
    qa_prompt_tmpl: str = DEFAULT_PROMPT_TMPL
    
    class Config:
        arbitrary_types_allowed = True
        
    @classmethod
    def from_defaults(
        cls,
        name: Optional[str] = None,
        st_memory: Optional[deque] = None,
        lt_memory: Optional[BaseGPTIndex] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs
    ) -> "ConvoAgent":
        name = name or "Agent"
        st_memory = st_memory or deque()
        lt_memory = lt_memory or GPTSimpleVectorIndex([], service_context=service_context)
        service_context = service_context or ServiceContext.from_defaults()
        return cls(
            name=name,
            st_memory=st_memory,
            lt_memory=lt_memory,
            service_context=service_context,
            **kwargs
        )
                      
    
    def add_message(self, message: str, user: str) -> None:
        """Add message from another user."""
        fmt_message = format_text(message, user)
        self.st_memory.append(fmt_message) 
        while len(self.st_memory) > self.st_memory_size:
            self.st_memory.popleft()
        self.lt_memory.insert(Document(fmt_message))
    
    def generate_message(self, prev_message: Optional[str] = None) -> str:
        """Generate a new message."""
        # if prev_message is None, get previous message using short-term memory
        if prev_message is None:
            prev_message = self.st_memory[-1]

        st_memory_text = "\n".join([l for l in self.st_memory])
        summary_response = self.lt_memory.query(
            f"Tell me a bit more about any context that's relevant "
            f"to the current messages: \n{st_memory_text}",
            response_mode="compact",
            **self.lt_memory_query_kwargs
        )
        
        # add both the long-term memory summary and the short-term conversation
        list_builder = GPTListIndex.from_documents([], service_context=self.service_context)
        list_builder.insert_nodes([Node(str(summary_response))])
        list_builder.insert_nodes([Node(st_memory_text)])
        
        # question-answer prompt
        full_qa_prompt_tmpl = (
            self.user_prefix_tmpl.format(name=self.name) + "\n" +
            self.qa_prompt_tmpl
        )
        qa_prompt = QuestionAnswerPrompt(full_qa_prompt_tmpl)
        
        response = list_builder.query(
            "Generate the next message in the conversation.", 
            text_qa_template=qa_prompt, 
            response_mode="compact"
        )    
        return str(response)

def run_conversation_loop(
    service_context,
    a_agent: ConvoAgent, 
    b_agent: ConvoAgent, 
    a_starter: Optional[str] = None, 
    b_starter: Optional[str] = None,
    num_iterations: Optional[int] = 100,
    a_alias = None,
    b_alias = None
) -> None:
    """Run conversation loop."""
    a_starter = a_starter or f"Hi, my name is {a_alias}!"
    b_starter = b_starter or f"Hi, my name is {b_alias}!" 

    a_agent.add_message(a_starter, a_alias)
    b_agent.add_message(a_starter, a_alias)

    a_agent.add_message(b_starter, b_alias)
    b_agent.add_message(b_starter, b_alias)
    
    # run conversation loop
    current_user = a_alias
    for _ in range(num_iterations):

        agent = a_agent if current_user == a_alias else b_agent
        new_message = agent.generate_message(service_context)

        message_to_print = format_text(new_message, current_user)
        print(message_to_print)

        a_agent.add_message(new_message, current_user)
        b_agent.add_message(new_message, current_user)

        current_user = a_alias if current_user == b_alias else b_alias


def get_service_context(model_path):    
    embeddings = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(embeddings)
    llm = LlamaCpp(max_tokens=150, n_ctx=512, model_path=model_path, temperature=0.8, n_threads=6, f16_kv=True, use_mlock=True)
    max_input_size = 125
    num_output = 30
    max_chunk_overlap = -100000
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
    return service_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    parser.add_argument('-a', type=str, required=False, default='Alice')
    parser.add_argument('-b', type=str, required=False, default='Bob')
    parser.add_argument('-i', type=str, required=False, default=100)
    parser.add_argument('-activity', type=str, required=False, default='on a first date')
    parser.add_argument('-a_starter', type=str, required=False)
    parser.add_argument('-b_starter', type=str, required=False)
    args = parser.parse_args()
    service_context = get_service_context(args.m)
    
    a_user_prefix_tmpl = (
        f"Your name is {args.a}. "
        f"You are {args.activity} with another person named {args.b}. "
        f"We provide conversation context between you and {args.b} below. "
    )
    b_user_prefix_tmpl = (
        f"Your name is {args.b}. "
        f"You are {args.activity} with another person named {args.a}. "
        f"We provide conversation context between you and {args.a} below. "
    )
    a_agent = ConvoAgent.from_defaults(
        name=args.a, 
        service_context=service_context, 
        user_prefix_tmpl=a_user_prefix_tmpl,
        lt_memory=GPTListIndex.from_documents([], service_context=service_context)
        
    )
    b_agent = ConvoAgent.from_defaults(
        name=args.b, 
        service_context=service_context,
        user_prefix_tmpl=b_user_prefix_tmpl,
        lt_memory=GPTListIndex.from_documents([], service_context=service_context)
    )
    run_conversation_loop(a_agent, b_agent, a_starter=args.a_starter, b_starter=args.b_starter, 
                          num_iterations=args.i, a_alias=args.a, b_alias=args.b)

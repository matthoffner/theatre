import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index import GPTListIndex
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from main import ConvoAgent, format_text, get_service_context

import asyncio
import uvicorn



logger = logging.getLogger()

MESSAGE_STREAM_DELAY = .5  # second
MESSAGE_STREAM_RETRY_TIMEOUT = 15000  # milisecond
app = FastAPI()

app.mount("/js", StaticFiles(directory="js"), name="js")

# add CORS so our web page can connect to our api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COUNTER = 0

class LlamaArgs:
    a = "Alice"
    b = "Ben"
    i = 1000
    activity = "creating a video game"
    model_path = ""
    a_starter = ""
    b_starter = ""

args = LlamaArgs()

def get_message(service_context):
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
    a_starter = args.a_starter or f"Hi, my name is {args.a}!"
    b_starter = args.b_starter or f"Hi, my name is {args.b}!" 

    a_agent.add_message(a_starter, args.a)
    b_agent.add_message(a_starter, args.a)

    a_agent.add_message(b_starter, args.b)
    b_agent.add_message(b_starter, args.b)
    
    # run conversation loop
    current_user = args.a
    message = []
    for _ in range(args.i):

        agent = a_agent if current_user == args.a else b_agent
        new_message = agent.generate_message(service_context)

        message_to_print = format_text(new_message, current_user)
        print(message_to_print)

        a_agent.add_message(new_message, current_user)
        b_agent.add_message(new_message, current_user)

        current_user = args.a if current_user == args.b else args.b
        message.append(message_to_print)
    
    return message


@app.get("/")
async def index(request: Request):
    # prompt = request.query_params.get('prompt')
    html_content = f"""
    <html>
        <head>
            <title>theatre</title>
        </head>
        <body>
            <h1 style="text-align:center;font-family:Arial">theatre</h1>
            <div id="root"></div>
            <script type="module" src="/js/index.js"></script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/stream")
async def message_stream(request: Request):
    service_context = get_service_context(model_path=args.model_path)
    async def event_generator(service_context):
        while True:
            if await request.is_disconnected():
                logger.debug("Request disconnected")
                break

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
            a_starter = args.a_starter or f"Hi, my name is {args.a}!"
            b_starter = args.b_starter or f"Hi, my name is {args.b}!" 

            a_agent.add_message(a_starter, args.a)
            b_agent.add_message(a_starter, args.a)

            a_agent.add_message(b_starter, args.b)
            b_agent.add_message(b_starter, args.b)
            
            # run conversation loop
            current_user = args.a
            for _ in range(args.i):
                agent = a_agent if current_user == args.a else b_agent
                new_message = agent.generate_message(service_context)

                message_to_print = format_text(new_message, current_user)
                print(message_to_print)

                a_agent.add_message(new_message, current_user)
                b_agent.add_message(new_message, current_user)

                current_user = args.a if current_user == args.b else args.b

                yield {
                    "event": "new_message",
                    "id": "message_id",
                    "retry": MESSAGE_STREAM_RETRY_TIMEOUT,
                    "data": message_to_print,
                }

                await asyncio.sleep(MESSAGE_STREAM_DELAY)

    return EventSourceResponse(event_generator(service_context))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

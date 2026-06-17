import os
from dotenv import load_dotenv


def init_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    required = ["OPENAI_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    return {name: os.getenv(name) for name in required}


if __name__ == "__main__":
    env = init_env()
    print("Environment initialized.")


from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "mymodel"


@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 199.99,
    }
    print(f"Executing get_product_price for {product}...")
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price."""
    
    discou_percentages = {
        
        "silver": 12,
        "gold": 23,
        "bronze": 5,
    }
    discount = discou_percentages.get(discount_tier, 0) / 100
    print(f"Executing apply_discount for price {price} with discount {discount}...")
    return round(price * (1 - discount), 2)
# the run agent function
def run_agent(question: str):

    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    messages = [
        SystemMessage(content="You are a helpful assistant that provides product pricing information and applies discounts." \
        "STRICT RULES: you must follow those exactly:" \
        " 1) ever guess or assume product prices, use the get_product_price tool." \
        " 2) nly call apply_discount after you have received the price from get_product_price. pass the exact price. Do not pass a made-up number."
        " 3)Always respond with the final answer after using the tools."), 


        HumanMessage(content=question)
    ]
    
    for _ in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {_}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        # if no tool calls this is the final answer
        if not tool_calls:
                print(f"Final answer: {ai_message.content}")
                return ai_message.content
        response = llm.invoke(messages)

        #process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            print(f"Tool {tool_name} not found. Skipping tool call.")
            continue
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")
        
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))   
    print("Max iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Running agent...")

    result = run_agent("What is the price of a laptop with a gold discount?")
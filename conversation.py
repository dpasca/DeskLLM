import json
from openai import OpenAI
from typing import List, Dict, Any, Callable

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_KEY = "ollama"
OLLAMA_DEFAULT_MODEL = "llama3.1"

# Global variable to hold the logging function
_log_lev_fn = None
# Define log functions outside the class
def log(message: str):
    if _log_lev_fn:
        _log_lev_fn("i", message)

def log_wrn(message: str):
    if _log_lev_fn:
        _log_lev_fn("w", message)

def log_err(message: str):
    if _log_lev_fn:
        _log_lev_fn("e", message)

class Conversation:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL + "/v1",
        api_key: str = OLLAMA_API_KEY,
        model: str = OLLAMA_DEFAULT_MODEL,
        system_prompt: str = "You are a highly capable and proactive AI assistant.",
        tools_get_descriptions_fn: Callable[[], List[Dict[str, Any]]] = None,
        tools_execute_tool_fn: Callable[[str, Dict[str, Any]], Any] = None,
        log_lev_fn: Callable[[str, str], None] = None):

        global _log_lev_fn
        _log_lev_fn = log_lev_fn or (lambda level, msg: None)

        log(f"Model: {model}, Base URL: {base_url}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.tools_get_descriptions_fn = tools_get_descriptions_fn or (lambda: [])
        self.tools_execute_tool_fn = tools_execute_tool_fn or (lambda tool_name, args: None)

        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def apply_tools(self, response, app_state):
        tc_reqs = []
        tools_out = []

        if not(hasattr(response, 'tool_calls') and response.tool_calls):
            log("No tool calls in the response")
            return tc_reqs, tools_out

        log(f"We have {len(response.tool_calls)} tool calls in the response")
        log(str(response.tool_calls))
        for call in response.tool_calls:
            tool_name = call.function.name
            args_str = call.function.arguments
            log(f"Raw arguments string: {args_str}")

            try:
                args = json.loads(call.function.arguments) if call.function.arguments else {}
            except:
                log_err(f"Tool call with invalid arguments: {call}")
                continue

            log(f"Executing tool {tool_name} with args {args}")
            try:
                result = self.tools_execute_tool_fn(tool_name, args)
                log(f"Raw execution result: {result}")
            except Exception as e:
                log_err(f"Error executing tool {tool_name}: {str(e)}")
                #result.response = f"Error executing tool {tool_name}: {str(e)}"
                # Create a fake result
                result = {"response": f"Error executing tool {tool_name}: {str(e)}"}
                #continue

            log(f"Processed Tool Call Result: {result}")

            # Append to list of requests
            tc_reqs.append({
                "id": call.id,
                "function": {
                    "name": tool_name,
                    "arguments": args_str,
                },
                "type": "function",
            })

            # Append to list of tool outputs
            content = None
            try:
                content = json.dumps(result)
            except:
                content = getattr(result, 'response', 'No response')

            tools_out.append({
                "tool_call_id": call.id,
                "role": "tool",
                "name": tool_name,
                "content": content
            })

        return tc_reqs, tools_out

    def get_assistant_response(self, app_state):
        log("Calling the completion...")
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools_get_descriptions_fn(),
            tool_choice="auto"
        )
        response = completion.choices[0].message

        log(f"Raw response: {response}")

        # Only append the assistant's response if content is not null
        if response.content:
            self.messages.append({"role": "assistant", "content": response.content})
        else:
            log_wrn("Assistant response content is null. Skipping append.")

        tc_reqs, tools_out = self.apply_tools(response, app_state)

        if tc_reqs and tools_out:
            log("Calling the completion after the tool call")
            log(f"tc_reqs: {tc_reqs}")
            log(f"tools_out: {tools_out}")

            # Build the message that details the requested tool calls
            self.messages.append({"role": "assistant", "tool_calls": tc_reqs})
            # Add the tools output right below the request message
            self.messages += tools_out

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            response = completion.choices[0].message

            # Only append the final assistant response if content is not null
            if response.content:
                self.messages.append({"role": "assistant", "content": response.content})
            else:
                log_wrn("Final assistant response content is null. Skipping append.")

        return response.content

#==================================================================
# Test the Conversation class
#==================================================================
if __name__ == "__main__":
    def main():
        global _log_lev_fn
        _log_lev_fn = lambda level, msg: print(f"[{level}] {msg}")

        def get_ollama_models():
            import requests
            try:
                response = requests.get(OLLAMA_BASE_URL + "/api/tags")
                if response.status_code == 200:
                    models = response.json()["models"]
                    return [model["name"] for model in models]
                else:
                    print(f"Error: Unable to fetch models. Status code: {response.status_code}")
                    return []
            except requests.RequestException as e:
                print(f"Error connecting to Ollama: {e}")
                return []

        def select_suitable_model(models):
            preferred_models = [OLLAMA_DEFAULT_MODEL, "qwen2.5"]
            for model in preferred_models:
                if any(model in m for m in models):
                    return next(m for m in models if model in m)
            return models[0] if models else None

        if ollama_models := get_ollama_models():
            suitable_model = select_suitable_model(ollama_models)
            print(f"Available Ollama models: {ollama_models}")
            print(f"Selected model: {suitable_model}")
        else:
            print("No Ollama models found.")
            return

        conv = Conversation(model=suitable_model)
        conv.add_user_message("Hello, how are you?")
        print(conv.get_assistant_response(None))

    main()

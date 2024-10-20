#==================================================================
# dyna_tools_manager.py
#
# Author: Davide Pasca, 2024/10/20
#==================================================================
import os
import json
import importlib.util
import time
import io
import sys
import threading
from typing import Dict, Any, List
import subprocess

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

class CodeExecutor(threading.Thread):
    """A class to execute Python code in a separate thread."""
    def __init__(self, code, globals_dict):
        threading.Thread.__init__(self)
        self.code = code
        self.globals_dict = globals_dict
        self.local_vars = {}
        self.result = None
        self.output = None
        self.error = None

    def run(self):
        try:
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Wrap the code in a function to handle return statements
            wrapped_code = f"def __temp_function():\n    " + "\n    ".join(self.code.splitlines())
            exec(wrapped_code, self.globals_dict, self.local_vars)
            self.result = self.local_vars['__temp_function']()

            self.output = captured_output.getvalue()
        except Exception as e:
            self.error = str(e)
        finally:
            sys.stdout = sys.__stdout__

class DynaToolsManager:
    def __init__(
            self,
            tools_dir='tools',
            debug=False,
            log_lev_fn=None,
            ):
        global _log_lev_fn
        _log_lev_fn = log_lev_fn or (lambda level, msg: None)
        self.tools_dir = tools_dir
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.debug = debug
        self.load_tool_definitions()
        self.create_add_tool_function()
        self.create_execute_python_function()

    @staticmethod
    def get_system_prompt() -> str:
        return ""

    def load_tool_definitions(self):
        # Create the tools directory if it doesn't exist
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir)
        # Load tool definitions from JSON files in the tools directory
        has_found_tools = False
        for filename in os.listdir(self.tools_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.tools_dir, filename)
                self.load_tool_definition(file_path)
                has_found_tools = True

        if not has_found_tools:
            canonical_path = os.path.abspath(self.tools_dir)
            log_wrn(f"No tools found in the tools directory: {canonical_path}")

    def load_tool_definition(self, json_file_path: str):
        log(f"Loading tool definition from {json_file_path}")
        try:
            with open(json_file_path, 'r') as file:
                tool_data = json.load(file)
        except json.JSONDecodeError:
            log_err(f"Error: Invalid JSON in {json_file_path}")
            return
        except Exception as e:
            log_err(f"Error loading {json_file_path}: {str(e)}")
            return

        tool_name = tool_data.get('name')
        source_file = tool_data.get('source')

        if not tool_name or not source_file:
            log_wrn(f"Invalid tool definition in {json_file_path}")
            return

        # Check and install required modules
        required_modules = tool_data.get('required_modules', [])
        self.check_and_install_modules(required_modules)

        # Add the tool definition to the tools dictionary
        self.tools[tool_name] = {
            'source_file': os.path.join(self.tools_dir, source_file),
            'description': tool_data.get('description', ''),
            'parameters': tool_data.get('parameters', {}),
            'creation_timestamp': tool_data.get('creation_timestamp', int(time.time())),
            'module': None,
            'function': None
        }

    def check_and_install_modules(self, required_modules: List[str]):
        for module in required_modules:
            try:
                importlib.import_module(module)
                log(f"Module '{module}' is already installed.")
            except ImportError:
                log_wrn(f"Module '{module}' is not installed. Attempting to install...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", module])
                    log(f"Successfully installed module '{module}'.")
                except subprocess.CalledProcessError as e:
                    log_err(f"Failed to install module '{module}'. Error: {str(e)}")

    def load_tool_module(self, tool_name: str):
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        if tool['module'] is not None:
            return  # Module already loaded

        source_path = tool['source_file']
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file {source_path} not found for tool {tool_name}")

        spec = importlib.util.spec_from_file_location(tool_name, source_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Error loading module for {tool_name}: {str(e)}")

        tool_function = getattr(module, tool_name, None)
        if not tool_function:
            raise AttributeError(f"Could not find function {tool_name} in {source_path}")

        tool['module'] = module
        tool['function'] = tool_function
    def get_tool(self, tool_name: str) -> Dict[str, Any]:
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_description(self, tool_name: str) -> str:
        tool = self.get_tool(tool_name)
        return tool['description'] if tool else f"Tool '{tool_name}' not found"

    def get_tool_parameters(self, tool_name: str) -> Dict[str, Any]:
        tool = self.get_tool(tool_name)
        return tool['parameters'] if tool else {}

    def get_tool_creation_time(self, tool_name: str) -> int:
        tool = self.get_tool(tool_name)
        return tool['creation_timestamp'] if tool else 0

    def get_descriptions(self) -> List[Dict[str, Any]]:
        """Get a list of tool descriptions in a format suitable for API consumption."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for name, tool in self.tools.items()
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        log(f"Executing tool '{tool_name}' with parameters: {kwargs}")
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        if tool['function'] is None:
            self.load_tool_module(tool_name)

        try:
            out = tool['function'](**kwargs)
            log(f"Tool output: {out}")
            return out
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{tool_name}': {str(e)}")

    # Update the reload_tools method
    def reload_tools(self):
        self.tools.clear()
        self.load_tool_definitions()

    # Update the add_tool method
    def add_tool(self, json_description: str, python_code: str) -> bool:
        try:
            tool_data = json.loads(json_description)
        except json.JSONDecodeError:
            log_err("Invalid JSON description")
            return False

        tool_name = tool_data.get('name')
        if not tool_name:
            log_err("Tool name not specified in JSON description")
            return False

        if tool_name in self.tools:
            log_err(f"Tool '{tool_name}' already exists")
            return False

        json_filename = f"{tool_name}.json"
        py_filename = f"{tool_name}.py"
        json_path = os.path.join(self.tools_dir, json_filename)
        py_path = os.path.join(self.tools_dir, py_filename)

        # Add creation timestamp and source file to the JSON description
        tool_data['creation_timestamp'] = int(time.time())
        tool_data['source'] = py_filename

        # Write JSON file
        try:
            with open(json_path, 'w') as json_file:
                json.dump(tool_data, json_file, indent=4)
        except Exception as e:
            log_err(f"Error writing JSON file: {str(e)}")
            return False

        # Write Python file
        try:
            with open(py_path, 'w') as py_file:
                py_file.write(python_code)
        except Exception as e:
            log_err(f"Error writing Python file: {str(e)}")
            os.remove(json_path)  # Remove the JSON file if Python file write fails
            return False

        # Update the local tools dictionary with just the definition
        tool_data = json.loads(json_description)
        tool_name = tool_data['name']
        self.tools[tool_name] = {
            'source_file': os.path.join(self.tools_dir, f"{tool_name}.py"),
            'description': tool_data.get('description', ''),
            'parameters': tool_data.get('parameters', {}),
            'creation_timestamp': int(time.time()),
            'module': None,
            'function': None
        }

        log(f"Tool '{tool_name}' added successfully")
        return True

    # Remove a tool from the tools directory
    def remove_tool(self, tool_name: str) -> bool:
        if tool_name not in self.tools:
            log_err(f"Tool '{tool_name}' not found")
            return False

        json_filename = f"{tool_name}.json"
        py_filename = f"{tool_name}.py"
        json_path = os.path.join(self.tools_dir, json_filename)
        py_path = os.path.join(self.tools_dir, py_filename)

        try:
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists(py_path):
                os.remove(py_path)
            del self.tools[tool_name]
            log(f"Tool '{tool_name}' removed successfully")
            return True
        except Exception as e:
            log_err(f"Error removing tool '{tool_name}': {str(e)}")
            return False

    # Create a special tool to call the add_tool method.
    def create_add_tool_function(self):
        tool_name = "add_tool"
        tool_description = "Add a new tool to the DynaToolsManager"
        tool_parameters = {
            "type": "object",
            "properties": {
                "json_description": {
                    "type": "string",
                    "description": "JSON string describing the new tool"
                },
                "python_code": {
                    "type": "string",
                    "description": "Python code implementing the new tool"
                }
            },
            "required": ["json_description", "python_code"]
        }

        # Add the tool to the tools dictionary
        self.tools[tool_name] = {
            'function': self.add_tool,
            'description': tool_description,
            'parameters': tool_parameters,
            'creation_timestamp': int(time.time())
        }

    # Execute Python code provided as a plain string.
    def execute_python(self, source_code: str, timeout: int = 5) -> str:
        log("Received code input for execution.")
        log(f"Code to execute: {repr(source_code)}")

        if not source_code:
            log("Code is empty.")
            return "Error: Code is empty. Please provide valid Python code."

        try:
            executor = CodeExecutor(source_code, globals())
            executor.start()
            executor.join(timeout)
        except Exception as e:
            log_err(f"Exception during code execution setup: {str(e)}")
            return f"Error: Exception during code execution setup: {str(e)}"

        if executor.is_alive():
            # Timeout case
            log_err("Code execution timed out.")
            return "Error: Code execution timed out."
        elif executor.error:
            # Error case
            log_err(f"Error during code execution: {executor.error}")
            return f"Error: Error during code execution: {executor.error}"
        else:
            log(f"Code executed successfully. Output: {executor.output}, Result: {executor.result}")
            # Where did we get our output ("output" or "result") ?
            if executor.result is not None:
                return str(executor.result)
            elif executor.output:
                return executor.output.strip()
            else:
                return ""

    # Create a special tool to call the execute_python method.
    def create_execute_python_function(self):
        tool_name = "execute_python"
        tool_description = """
Execute Python code for any programming task, including system administration, data analysis, and more.
Provide source code as a JSON-escaped string.
Code should not be part of a JSON object.
Any output must be produced via the print() function.
"""
        tool_parameters = {
            "type": "object",
            "properties": {
                "source_code": {
                    "type": "string",
                    "description": "JSON-escaped string containing the Python code to execute."
                }
            },
            "required": ["source_code"]
        }

        self.tools[tool_name] = {
            'function': self.execute_python,
            'description': tool_description,
            'parameters': tool_parameters,
            'creation_timestamp': int(time.time())
        }

#==================================================================
# Usage example:
#==================================================================
if __name__ == "__main__":
    def main():
        def log_lev(level="i", msg=""):
            import inspect
            level_str = "[ERR]" if level == "e" else "[WARN]" if level == "w" else ""
            # Get the current stack
            stack = inspect.stack()
            # Find the first frame that doesn't belong to a logging function
            caller_frame = None
            for frame in stack[1:]:  # Skip the current frame
                if not frame.function.startswith(('log', 'log_err', 'log_wrn')):
                    caller_frame = frame
                    break
            func_str = f"[{caller_frame.function}]" if caller_frame else "[Unknown]"
            print(f"{func_str}{level_str} {msg}")

        manager = DynaToolsManager(debug=True, log_lev_fn=log_lev)
        log("Available tools: " + ", ".join(manager.list_tools()))

        for tool_name in manager.list_tools():
            creation_time = manager.get_tool_creation_time(tool_name)
            log(f"Tool '{tool_name}' was created at Unix timestamp: {creation_time}")

        # Example of getting tool descriptions
        descriptions = manager.get_descriptions()
        log("Tool descriptions:")
        for desc in descriptions:
            log(json.dumps(desc, indent=2))

        # Example of using the new add_tool function to create a "hello_world" tool
        json_description = '''
{
    "name": "hello_world",
    "description": "A simple tool that says hello",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the person to greet"
            }
        },
        "required": ["name"]
    }
}'''

        python_code = '''
def hello_world(name: str) -> str:
    return f"Hello, {name}! Welcome to the world of dynamic tools."
'''

        success = manager.execute_tool("add_tool", json_description=json_description, python_code=python_code)
        if success:
            log("Hello World tool added successfully")
            # Test the new tool
            result = manager.execute_tool("hello_world", name="Alice")
            log(result)
        else:
            log_err("Failed to add Hello World tool")

        # Display updated list of tools
        log("Updated available tools: " + ", ".join(manager.list_tools()))

        # Remove the hello_world tool
        if manager.remove_tool("hello_world"):
            log("Hello World tool removed successfully")
        else:
            log_err("Failed to remove Hello World tool")

        # Display final list of tools
        log("Final available tools: " + ", ".join(manager.list_tools()))

    main()


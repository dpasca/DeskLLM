import threading
from typing import List, Tuple
from imgui_bundle import hello_imgui, imgui, immapp, ImVec2, ImVec4
from imgui_bundle import imgui_md
from dyna_tools_manager import DynaToolsManager
from conversation import Conversation
import inspect

ENABLE_TOOL_CALLS = True

_app_state = None

# Define a function to append logs with a maximum limit
def append_log(message):
    global _app_state
    _app_state.log_messages.append(message)
    MAX_LOG_MESSAGES = 1000  # Limit to 1000 messages
    if len(_app_state.log_messages) > MAX_LOG_MESSAGES:
        _app_state.log_messages.pop(0)

def make_nice_log_str(level, message):
    if not isinstance(message, str):
        message = str(message)

    # Get the entire call stack
    stack = inspect.stack()

    # Find the most relevant caller
    caller_info = None
    for frame_info in stack[1:]:  # Skip the current function
        if frame_info.function not in ['log', 'log_wrn', 'log_err', 'log_lev', '<lambda>'] and frame_info.filename != '<string>':
            caller_info = frame_info
            break

    if caller_info:
        frame = caller_info.frame
        module = inspect.getmodule(frame)
        class_name = None

        # Check if it's a method of a class
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__

        # Get the function name
        func_name = caller_info.function

        # Construct the full caller name
        if class_name:
            caller_name = f"{class_name}.{func_name}"
        elif module and module.__name__ != '__main__':
            caller_name = f"{module.__name__}.{func_name}"
        else:
            caller_name = func_name
    else:
        caller_name = "Unknown"

    # If caller_name is still __main__ or <lambda>, use a more generic name
    if caller_name in ['__main__', '<lambda>', '__main__.<lambda>']:
        caller_name = "App"

    level_str = "[ERR]" if level == "e" else "[WARN]" if level == "w" else ""
    out_str = f"{level_str}[{caller_name}] {message}"
    return out_str

def log_lev(level, message):
    out_str = make_nice_log_str(level, message)
    append_log(out_str)
    print(out_str)  # Also print to console

def log(message):
    log_lev("i", message)

def log_wrn(message):
    log_lev("w", message)

def log_err(message):
    log_lev("e", message)

#===============================================================================
SAMPLE_CHAT_HISTORY = [
    ("user", "Hello, how are you?"),
    ("assistant", "I'm doing great, thanks for asking!"),
    ("user", "Please show me sample Markdown formatting, including a simple table."),
    ("assistant", """Here some formatted text:

## Header 2
### Header 3
**bold**
*italic*
[link](https://www.example.com)
![image](https://www.example.com/image.png)

Here is a simple table:

| Header 1 | Header 2 | Header 3 |
| -------- | -------- | -------- |
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
| Cell 7   | Cell 8   | Cell 9   |

Here is a code block:

```python
print('Hello, world!')
```""")
]

#===============================================================================
# AppState
#===============================================================================
class AppState:
    def __init__(self, model):
        # Singleton 8)
        global _app_state
        _app_state = self

        base_url = None
        api_key = None

        if not model.startswith("gpt"):
            base_url = "http://localhost:11434/v1"
            api_key = "ollama"

        self.log_messages: List[str] = []  # Log buffer

        self.tools_mng = None
        if ENABLE_TOOL_CALLS:
            self.tools_mng = DynaToolsManager(
                debug=True,
                log_lev_fn=lambda level, msg: log_lev(level, msg)
            )

        # Common prompt
        base_prompt = """
You are a highly intelligent and and friendly AI assistant.
Help the user as much as you can.
"""
        if ENABLE_TOOL_CALLS:
            # Prompt for tool calls
            base_prompt += """
Key guidelines:
1. Utilize the tools available to you to answer the user queries.
2. Be proactive: your goal is to minimize user effort.
3. Show your work: Before providing your final answer, explain your thought process using <thinking> tags.
4. Be creative: Approach problems from multiple angles and consider related topics that might be useful.

Remember: Your primary focus is on solving problems efficiently using the tools available to you.
Don't ask the user to perform tasks that you can do yourself using the provided tools.
"""
        else:
            # Prompt for no tool calls
            base_prompt += """
Key guidelines:
1. Be proactive: your goal is to minimize user effort.
2. Show your work: Before providing your final answer, explain your thought process using <thinking> tags.
3. Be creative: Approach problems from multiple angles and consider related topics that might be useful.
"""

        tc_get_descs = lambda: (
            self.tools_mng.get_descriptions()
            if ENABLE_TOOL_CALLS
            else []
        )
        tc_execute_tool = lambda tool_name, args: (
            self.tools_mng.execute_tool(tool_name, **args)
            if ENABLE_TOOL_CALLS
            else None
        )

        self.conv = Conversation(
            base_url=base_url,
            api_key=api_key,
            model=model,
            # The system prompt is the base prompt + the tools descriptions
            system_prompt=base_prompt + DynaToolsManager.get_system_prompt(),
            # Callbacks to DynaToolsManager to get tool descriptions and execute tools
            tools_get_descriptions_fn=tc_get_descs,
            tools_execute_tool_fn=tc_execute_tool,
            # Callbacks to Conversation to log messages
            log_lev_fn=lambda level, msg: log_lev(level, msg)
        )
        self.chat_history: List[Tuple[str, str]] = []  # List of tuples (message_type, message_content)
        self.user_input = ""

        # Use sample chat history for testing
        #self.chat_history = SAMPLE_CHAT_HISTORY

        # Keep track of what's being edited
        self.edit_content = None
        self.edit_index = -1

def send_message(app_state):
    if app_state.user_input.strip():
        user_input = app_state.user_input.strip()
        app_state.chat_history.append(("user", user_input))  # Append as a user message
        # Clear the input
        app_state.user_input = ""

        # Start a thread to get the assistant's response
        threading.Thread(target=get_assistant_response_thread, args=(app_state, user_input)).start()

def get_assistant_response_thread(app_state, user_input):
    app_state.chat_history.append(("assistant", "..."))  # Placeholder for assistant response

    try:
        # Send message to assistant
        app_state.conv.add_user_message(user_input)
        response = app_state.conv.get_assistant_response(app_state)
        # Remove the placeholder
        app_state.chat_history.pop()
        # Append assistant response to chat history
        app_state.chat_history.append(("assistant", response))
        log("Assistant responded successfully.")
    except Exception as e:
        # Remove the placeholder
        app_state.chat_history.pop()
        # Append error message as assistant message
        app_state.chat_history.append(("assistant", f"[Error] {str(e)}"))
        log_err(f"Assistant error: {str(e)}")

#===============================================================================
# GUI Edit
#===============================================================================
def gui_on_edit_message(app_state, index, content):
    if app_state.edit_index == index:
        app_state.edit_content = None
        app_state.edit_index = -1
    else:
        app_state.edit_content = content
        app_state.edit_index = index

def gui_edit_window(app_state):
    index = app_state.edit_index
    if index == -1:
        return

    # Get the main viewport size
    viewport_size = imgui.get_main_viewport().size

    # Calculate 70% of the viewport size
    window_width = int(viewport_size.x * 0.7)
    window_height = int(viewport_size.y * 0.7)

    # Set the initial window size and position
    imgui.set_next_window_size(ImVec2(window_width, window_height), imgui.Cond_.once)
    imgui.set_next_window_pos(
        ImVec2((viewport_size.x - window_width) / 2, (viewport_size.y - window_height) / 2),
        imgui.Cond_.once
    )
    # Begin the window
    imgui.begin(f"Edit Message", flags=imgui.WindowFlags_.no_collapse)

    # Edit mode
    changed, new_content = imgui.input_text_multiline(
        label=f"##editbox",
        str=app_state.edit_content,
        size=ImVec2(-20, -40)
    )
    if changed:
        app_state.edit_content = new_content

    if imgui.button(f"Save"):
        # Replace the content of the message
        app_state.chat_history[index] = (app_state.chat_history[index][0], app_state.edit_content)
        app_state.edit_content = None
        app_state.edit_index = -1

    imgui.same_line()

    if imgui.button(f"Cancel"):
        app_state.edit_content = None
        app_state.edit_index = -1

    imgui.end()

def gui_message_buttons(app_state, index, content):
    #if not (imgui.is_window_hovered() or app_state.edit_index == index):
    #    return

    # Get the available width of the content region
    available_width = imgui.get_content_region_avail().x

    # Calculate the total width of the buttons
    button_width = imgui.calc_text_size("Edit").x + imgui.calc_text_size("Copy").x + imgui.get_style().item_spacing.x * 3

    # Set the cursor position to the right side
    imgui.set_cursor_pos_x(available_width - button_width)

    # Draw the buttons
    if imgui.small_button(f"Edit##{index}"):
        gui_on_edit_message(app_state, index, content)

    imgui.same_line()

    if imgui.small_button(f"Copy##{index}"):
        imgui.set_clipboard_text(content)

#===============================================================================
# GUI Main
#===============================================================================
def gui_function(app_state):
    # Create the chat history area
    size = ImVec2(0, -50)  # Leave space for the input at the bottom
    imgui.begin_child("ChatHistory", size=size, child_flags=0)

    # Calculate the content width, subtracting the right padding
    RIGHT_PADDING = 6
    content_width = imgui.get_content_region_avail().x - RIGHT_PADDING

    # Empty message to start the chat history because 1st message takes a
    # screen worth of height for some reason
    imgui.begin_child(
        "Message_-1",
        size=ImVec2(content_width, 0),
        child_flags=imgui.ChildFlags_.always_use_window_padding
    )
    imgui.end_child()

    for index, message in enumerate(app_state.chat_history):
        msg_type, content = message

        # If in edit mode, use the edited content
        if app_state.edit_index == index:
            content = app_state.edit_content

        # Background color based on message type
        bg_color = ImVec4(0.2, 0.2, 0.2, 1.0) if msg_type == "user" else ImVec4(0.25, 0.25, 0.25, 1.0)
        imgui.push_style_color(imgui.Col_.child_bg, bg_color)

        imgui.begin_child(
            f"Message_{index}",
            size=ImVec2(content_width, 0),
            child_flags=(imgui.ChildFlags_.always_use_window_padding |
                         imgui.ChildFlags_.auto_resize_y)
        )

        # Render the message content
        imgui_md.render(content)

        # Call the new function for Edit and Copy buttons
        gui_message_buttons(app_state, index, content)

        imgui.end_child()

        # Pop the added style variables
        imgui.pop_style_color()

    # Auto scroll to bottom
    if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
        imgui.set_scroll_here_y(1.0)
    imgui.end_child()

    # Handle edit mode UI
    gui_edit_window(app_state)

    imgui.separator()

    # Input field
    imgui.push_item_width(-50)
    changed, app_state.user_input = imgui.input_text(
        "##UserInput",
        app_state.user_input,
        flags=imgui.InputTextFlags_.enter_returns_true
    )
    enter_pressed = changed and imgui.is_key_pressed(imgui.Key.enter)
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button("Send") or (enter_pressed and app_state.user_input.strip()):
        send_message(app_state)

#===============================================================================
# GUI Log Window
#===============================================================================
def log_window_gui(app_state):
    # Begin the log window
    imgui.begin("Logs")

    # Calculate available space for the child window
    available_width = imgui.get_content_region_avail().x
    available_height = imgui.get_content_region_avail().y

    # Create a child region to enable scrolling with border, using available space
    imgui.begin_child("LogChild", 
                      size=ImVec2(available_width, available_height), 
                      child_flags=imgui.ChildFlags_.border)

    # Display each log message with color coding based on log level
    for message in app_state.log_messages:
        if message.startswith("[ERR]"):
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0))  # Red
            imgui.text_wrapped(message)
            imgui.pop_style_color()
        elif message.startswith("[WRN]"):
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 0.0, 1.0))  # Yellow
            imgui.text_wrapped(message)
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.8, 0.8, 0.8, 1.0))  # Gray
            imgui.text_wrapped(message)
            imgui.pop_style_color()

    # Auto scroll to the bottom
    if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
        imgui.set_scroll_here_y(1.0)

    imgui.end_child()
    imgui.end()

def setup_docking(app_state):
    docking_params = hello_imgui.DockingParams()

    # Create docking splits: main chat area and log window at the bottom
    split_main_log = hello_imgui.DockingSplit()
    split_main_log.initial_dock = "MainDockSpace"
    split_main_log.new_dock = "LogSpace"
    split_main_log.direction = imgui.Dir.down
    split_main_log.ratio = 0.3  # Percentage of height for the log window

    docking_params.docking_splits = [split_main_log]

    # Define Dockable Windows
    # Main Chat Window
    main_chat_window = hello_imgui.DockableWindow()
    main_chat_window.label = "Chat"
    main_chat_window.dock_space_name = "MainDockSpace"
    main_chat_window.gui_function = lambda: gui_function(app_state)

    # Log Window
    log_window = hello_imgui.DockableWindow()
    log_window.label = "Logs"
    log_window.dock_space_name = "LogSpace"
    log_window.gui_function = lambda: log_window_gui(app_state)

    docking_params.dockable_windows = [main_chat_window, log_window]

    return docking_params

#===============================================================================
# Config
#===============================================================================
import json
def load_config():
    try:
        with open(".config.json", "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {
            "model": "qwen2.5",
            "font_scale": 1.0,
            "adjust_font_to_dpi": True
        }
        save_config(config)
    return config

def save_config(config):
    with open(".config.json", "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

#===============================================================================
# Main
#===============================================================================
def main():
    config = load_config()
    app_state = AppState(model=config.get("model", "qwen2.5"))

    # Initialize Markdown renderer
    imgui_md.initialize_markdown()

    # Set up the app
    rparams = hello_imgui.RunnerParams()

    # Add font loader function to the runner params, for Markdown rendering
    rparams.callbacks.load_additional_fonts = imgui_md.get_font_loader_function()
    # Set the window title
    rparams.app_window_params.window_title = "DeskLLM"

    # Set initial window size
    rparams.app_window_params.window_geometry.size = (1024, 1200)

    # Enable docking by setting the default ImGui window type to provide a full-screen dock space
    rparams.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space

    # Setup docking
    docking_params = setup_docking(app_state)
    rparams.docking_params = docking_params

    # Set the default layout condition to application start
    rparams.docking_params.layout_condition = hello_imgui.DockingLayoutCondition.application_start

    def post_init():
        io = imgui.get_io()
        # Set the global font scale
        io.font_global_scale *= config["font_scale"]

        if config.get("adjust_font_to_dpi", False):
            # Adjust DPI scaling without reloading fonts
            style = imgui.get_style()
            style.scale_all_sizes(1.0)

    rparams.callbacks.post_init = post_init

    # Run the app
    hello_imgui.run(rparams)

    # Cleanup Markdown renderer
    imgui_md.de_initialize_markdown()

if __name__ == "__main__":
    main()


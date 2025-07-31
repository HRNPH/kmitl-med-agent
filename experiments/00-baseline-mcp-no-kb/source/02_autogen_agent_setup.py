# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # AutoGen Agent Setup
#
# This notebook sets up the Microsoft AutoGen agents for the KMITL Medical Agent experiment.

# ## 1. AutoGen Configuration


def setup_autogen_config():
    """Setup AutoGen configuration with Ollama"""
    config_list = [
        {
            "model": OLLAMA_MODEL,
            "api_base": OLLAMA_API_URL,
            "api_type": "open_ai",
            "api_key": "dummy",  # Ollama doesn't require real API key
        }
    ]
    return config_list


# Create the configuration
config_list = setup_autogen_config()
print("✓ AutoGen configuration created")

# ## 2. Create AutoGen Agents


def setup_agents():
    """Setup AutoGen agents"""

    # Medical Assistant Agent
    medical_assistant = AssistantAgent(
        name="medical_assistant",
        system_message="""You are a medical assistant specialized in Thai healthcare system. 
        You help answer questions about:
        - Hospital departments and services
        - Patient rights and insurance coverage
        - Medical procedures and treatments
        - Emergency protocols
        - Healthcare policies and regulations
        
        Always provide accurate, helpful information in Thai or English as appropriate.
        If you need to access external data or tools, use the available MCP server connections.""",
        llm_config={
            "config_list": config_list,
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )

    # User Proxy Agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # Automated mode
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
        code_execution_config={"work_dir": "workspace"},
        llm_config={
            "config_list": config_list,
            "temperature": 0.7,
        },
    )

    return {"medical_assistant": medical_assistant, "user_proxy": user_proxy}


# Create the agents
agents = setup_agents()
print("✓ AutoGen agents created successfully")

# ## 3. Test Agent Communication


async def test_agent_communication():
    """Test communication between agents"""
    print("\nTesting agent communication...")

    user_proxy = agents["user_proxy"]
    medical_assistant = agents["medical_assistant"]

    # Test message
    test_question = "แผนกฉุกเฉินเปิดกี่โมง?"

    try:
        # Start conversation
        chat_history = await user_proxy.a_initiate_chat(
            medical_assistant, message=f"Test question: {test_question}", max_turns=2
        )

        if chat_history:
            print("✓ Agent communication successful")
            print(f"Response: {chat_history[-1]['content'][:200]}...")
            return True
        else:
            print("✗ No response from agents")
            return False

    except Exception as e:
        print(f"✗ Agent communication failed: {e}")
        return False


# Test the agents (commented out to avoid blocking)
# await test_agent_communication()

print("✓ AutoGen agent setup completed!")

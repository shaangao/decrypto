import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.role_client import RoleClient
from src.types import AnthropicModel

def test_anthropic_extended_thinking(max_tokens, budget_tokens):
    print(f"\n--- Testing Claude with max_tokens={max_tokens}, budget_tokens={budget_tokens} ---")
    
    # Load environment variables (API keys)
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        return False
        
    model = AnthropicModel(
        model_key="claude4.6",
        model_id="claude-sonnet-4-6",
        api_key_name="ANTHROPIC_API_KEY",
        temperature=0.0,
        max_tokens=max_tokens,
        max_reasoning_tokens=budget_tokens
    )
    
    try:
        # Initialize as interceptor since that's what fails in Gopnik/Piaget config
        client = RoleClient(
            role="interceptor",
            model=model,
            mode=0,
            no_error_history=True
        )
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return False
        
    # The actual Gopnik False Belief prompt generated from src/env.py
    # We call get_gopnik_prompt directly to mirror the exact string
    from src.env import get_gopnik_prompt
    
    info = {
        'turn': 3,
        'hint history': [['red', 'fruit', 'crisp'], ['yellow', 'long', 'peel'], ['red', 'pit', 'sweet']],
        'code_history': [[1,2,3], [1,2,3]]
    }
    keywords = ["apple", "banana", "cherry", "date"]
    
    prompt = get_gopnik_prompt(info, keywords, "false_belief")
    
    print(f"System Prompt Length: {len(client.system_prompt)} characters")
    print(f"User Prompt Length: {len(prompt)} characters")
    print("Calling chat_completion...")
    
    try:
        hints, raw_output, info = client.chat_completion(
            message=prompt,
            code=None,
            keywords=None,
            hints=['crisp', 'peel', 'sweet'],
            hint_history=info['hint history'],
            predict_code=False,
            predict_keywords=True # Gopnik prompt asks for keywords
        )
        
        print(f"\nAPI Call Succeeded!")
        print(f"Attempts taken: {info['attempts']}")
        print(f"Raw Output Length: {len(raw_output) if raw_output else 0} characters")
        
        if info['fail'] > 0:
            print("WARNING: Reached max attempts or failed to extract valid JSON.")
            return False
            
        return True
        
    except Exception as e:
        import traceback
        print(f"\nAPI Call FAILED with exception:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Anthropic API Tests with Actual Game Prompts...\n")
    
    # Test 1: The current failing configuration
    print("Test 1: Running with the currently failing token limits")
    test1_success = test_anthropic_extended_thinking(max_tokens=1500, budget_tokens=1024)
    
    # Test 2: The recommended safe configuration
    print("\nTest 2: Running with recommended safe token limits")
    test2_success = test_anthropic_extended_thinking(max_tokens=8000, budget_tokens=1024)
    
    print("\n--- Summary ---")
    print(f"Test 1 (1500/1024): {'SUCCESS' if test1_success else 'FAILED'}")
    print(f"Test 2 (8000/1024): {'SUCCESS' if test2_success else 'FAILED'}")
#!/usr/bin/env python3
"""
Test Ollama direct client connection
Uses the same import pattern as the updated agents
"""

from ollama import Client
import time

def test_simple():
    """Test 1: Simple hello query"""
    print("\n" + "=" * 70)
    print("TEST 1: SIMPLE HELLO QUERY")
    print("=" * 70)
    
    client = Client(host="http://127.0.0.1:11434")
    
    start = time.time()
    resp = client.generate(model="llama3.2:1b", prompt="Say hello from Ollama")
    elapsed = time.time() - start
    
    print(f"Response: {resp['response']}")
    print(f"Time: {elapsed:.2f}s")
    return True


def test_supplier_analysis():
    """Test 2: Supplier analysis query (similar to agent queries)"""
    print("\n" + "=" * 70)
    print("TEST 2: SUPPLIER ANALYSIS QUERY")
    print("=" * 70)
    
    query = """
    Analyze this supplier and provide a risk score (0-100):
    
    Supplier: Sunrise Sustainable
    Location: Morocco
    Product: Solar panels
    
    Key Facts:
    - Production capacity at 70% utilization
    - Seasonal demand fluctuates 50%
    - Located in region with flooding risk
    - 40% debt-to-equity ratio
    
    Provide:
    RISK_SCORE: [number]
    RISK_DETAILS: [2-3 sentence summary]
    RISK_FACTORS: [list key concerns]
    """
    
    print(f"Query length: {len(query)} chars")
    print("Sending to LLM...")
    
    client = Client(host="http://127.0.0.1:11434")
    
    start = time.time()
    resp = client.generate(model="llama3.2:1b", prompt=query)
    elapsed = time.time() - start
    
    response_text = resp['response']
    
    print("\nResponse:")
    print("-" * 70)
    print(response_text)
    print("-" * 70)
    print(f"\nResponse length: {len(response_text)} chars")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {len(response_text) / elapsed:.1f} chars/sec")
    
    return True


if __name__ == "__main__":
    print("\n🚀 Testing Ollama Direct Client Connection\n")
    
    try:
        # Test 1: Simple query
        test_simple()
        
        # Test 2: Complex supplier analysis
        test_supplier_analysis()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 This confirms the ollama.Client approach works correctly.")
        print("   Your agents should now use this same pattern.\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

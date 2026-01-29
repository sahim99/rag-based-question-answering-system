"""
E2E Test Script for RAG System
Tests upload, query, and edge cases
"""
import requests
import json
import time

API_URL = "http://127.0.0.1:8000/api"

def test_health():
    """Test 1: Health check"""
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("✅ Test 1: Health Check PASSED")
    return True

def test_upload_txt():
    """Test 2: Upload TXT file"""
    content = "This is a test document about Python programming. Python is a versatile language used for web development, data science, and AI."
    files = {"file": ("test_doc.txt", content, "text/plain")}
    r = requests.post(f"{API_URL}/upload", files=files, timeout=60)
    assert r.status_code == 200, f"TXT upload failed: {r.status_code} - {r.text}"
    print("✅ Test 2: TXT Upload PASSED")
    return True

def test_query_simple():
    """Test 3: Simple factual query"""
    payload = {"question": "What is Python used for?"}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    assert r.status_code == 200, f"Query failed: {r.status_code}"
    data = r.json()
    assert "answer" in data, "No answer in response"
    assert "sources" in data, "No sources in response"
    print(f"✅ Test 3: Simple Query PASSED (Answer length: {len(data['answer'])} chars)")
    return data

def test_query_summary():
    """Test 4: Summary query"""
    payload = {"question": "Give me a summary of the project"}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    assert r.status_code == 200, f"Summary query failed: {r.status_code}"
    data = r.json()
    assert len(data.get("answer", "")) > 20, "Answer too short"
    print(f"✅ Test 4: Summary Query PASSED (Sources: {len(data.get('sources', []))})")
    return data

def test_query_edge_typo():
    """Test 5: Query with typo"""
    payload = {"question": "teck stack"}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    assert r.status_code == 200, f"Typo query failed: {r.status_code}"
    print("✅ Test 5: Typo Handling PASSED")
    return True

def test_sources_validation(query_result):
    """Test 6: Validate sources structure"""
    sources = query_result.get("sources", [])
    if sources:
        for src in sources:
            assert "source_file" in src, "Missing source_file"
            assert "chunk_id" in src, "Missing chunk_id"
        print(f"✅ Test 6: Sources Validation PASSED ({len(sources)} sources)")
    else:
        print("⚠️ Test 6: No sources returned (may be expected)")
    return True

def run_all_tests():
    print("\n" + "="*50)
    print("RAG SYSTEM END-TO-END TESTS")
    print("="*50 + "\n")
    
    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    tests = [
        ("Health Check", test_health),
        ("TXT Upload", test_upload_txt),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            results["passed"] += 1
            results["tests"].append({"name": name, "status": "PASS"})
        except Exception as e:
            results["failed"] += 1
            results["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
            print(f"❌ {name} FAILED: {e}")
    
    # Query tests
    try:
        query_result = test_query_simple()
        results["passed"] += 1
        results["tests"].append({"name": "Simple Query", "status": "PASS"})
        
        test_sources_validation(query_result)
        results["passed"] += 1
        results["tests"].append({"name": "Sources Validation", "status": "PASS"})
    except Exception as e:
        results["failed"] += 1
        print(f"❌ Query Test FAILED: {e}")
    
    try:
        test_query_summary()
        results["passed"] += 1
        results["tests"].append({"name": "Summary Query", "status": "PASS"})
    except Exception as e:
        results["failed"] += 1
        print(f"❌ Summary Query FAILED: {e}")
    
    try:
        test_query_edge_typo()
        results["passed"] += 1
        results["tests"].append({"name": "Typo Handling", "status": "PASS"})
    except Exception as e:
        results["failed"] += 1
        print(f"❌ Typo Test FAILED: {e}")
    
    print("\n" + "="*50)
    print(f"RESULTS: {results['passed']} PASSED, {results['failed']} FAILED")
    print("="*50)
    
    return results

if __name__ == "__main__":
    run_all_tests()

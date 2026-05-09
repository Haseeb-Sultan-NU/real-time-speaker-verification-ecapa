import requests

BASE_URL = "http://127.0.0.1:8000"

def test_auth_flow():
    print("--- 1. Testing Signup ---")
    signup_payload = {
        "user_id": "demo_user",
        "email": "demo@awaaz.com",
        "password": "SuperSecretPassword123"
    }
    res = requests.post(f"{BASE_URL}/signup", json=signup_payload)
    print(f"Status: {res.status_code}")
    print(res.json())

    print("\n--- 2. Testing Login (Correct Password) ---")
    login_payload = {
        "email": "demo@awaaz.com",
        "password": "SuperSecretPassword123"
    }
    res = requests.post(f"{BASE_URL}/login", json=login_payload)
    print(f"Status: {res.status_code}")
    print(res.json())

    print("\n--- 3. Testing Login (Wrong Password) ---")
    bad_login = {
        "email": "demo@awaaz.com",
        "password": "WrongPassword!"
    }
    res = requests.post(f"{BASE_URL}/login", json=bad_login)
    print(f"Status: {res.status_code}")
    print(res.json())

if __name__ == "__main__":
    test_auth_flow()
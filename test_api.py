import requests
import json

API_KEY = 'sk-238317dcae0944479ccb345a4bb47cc6'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# 测试不同的 embedding 模型名
models = [
    'deepseek-embedding-v1',
    'deepseek-embedding',
    'text-embedding-ada-002',
    'text-embedding-3-small',
]

for model in models:
    print(f'测试模型: {model}')
    payload = {
        'input': 'hello',
        'model': model
    }
    try:
        response = requests.post('https://api.deepseek.com/v1/embeddings', headers=headers, json=payload, timeout=10)
        print(f'  Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'  Success! Dimension: {len(data["data"][0]["embedding"])}')
            break
        else:
            print(f'  Response: {response.text[:100]}')
    except Exception as e:
        print(f'  Error: {e}')

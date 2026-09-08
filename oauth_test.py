def test_for_oauth_token():
    with open("app.py", 'r', encoding='utf-8') as f:
        code = f.read()
        assert "gr.OAuthToken" in code

# Venice Travel Assistant with RAG Chat

A Streamlit travel assistant for Venice, Italy. Email/password sign-in uses Firebase Authentication. Answers use only the 18 entries in `knowledge_base.json`, with guide IDs and retrieved cosine scores displayed beside each answer. No database is used.

## Student information

- Name: Đặng Nguyễn Yến Phương
- Student ID: 25127230
- Class: 25C09
- Chosen city: Venice, Italy
- Demonstration video: https://drive.google.com/drive/folders/1LWZHxowaHF-pAPkYvJws7T4V-aC7queA?usp=sharing

## Setup

Use Python 3.12 (recommended). From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

1. Use your Lab 3 Firebase project or create one. Enable **Authentication > Sign-in method > Email/Password**.
2. If `.env` does not exist, copy `.env.example` to `.env`. Fill in `FIREBASE_API_KEY` from Firebase Project settings. `FIREBASE_PROJECT_ID` is optional reference information; the REST authentication calls use the API key.
3. Start the Ollama server and Pinggy tunnel from Lab 4. On the **Ollama server**, ensure both models are installed:

   ```bash
   ollama pull embeddinggemma
   ollama pull qwen3:4b
   ```

4. Set `OLLAMA_BASE_URL` in `.env` to the current Pinggy base URL, without `/api/chat` or `/api/embed`. Use the tunnel setup from Lab 4. When its URL changes, update `.env` and restart Streamlit. For local development with Ollama on this computer, `http://localhost:11434` can be used instead.
5. Run:

   ```bash
   streamlit run app.py
   ```

6. Create an account, then sign in. The first question prepares the guide embeddings and can take longer. Subsequent questions reuse them.

Never commit `.env`, passwords, tokens or service-account keys. `.env.example` deliberately contains empty values. Firestore, Google OAuth and Firebase Admin SDK are not needed.


## Test questions for the assignment

The Streamlit test run produced **5 correct answers out of 6**, including two correct refusals. The screenshots show the answers and their citations. The retrieval panels were collapsed, so the full top-k lists still need to be recorded.

 Type | Question | Expected entry ID / result | Retrieved IDs (cosine) | Observed answer | Correct? | Failure stage if wrong |
| --- | --- | --- | --- | --- | --- | --- |
| Answerable | What can I see at the Murano Glass Museum? | G05 | G05 confirmed by citation. Full ranking pending. | Ancient Roman glass, historical Murano pieces, filigree, millefiori, glass beads and chandeliers [G05]. | Yes | — |
| Answerable | Where is the Burano Lace Museum located? | G06 | G06 confirmed by citation. Full ranking pending. | Piazza Galuppi on Burano, an island in Venice [G06]. | Yes | — |
| Answerable | How do I reach the waterbus dock at Marco Polo Airport? | G13. Follow the blue Water Transport signs to the first floor, then take the elevated pedestrian route with moving walkways. | G13 (0.7184), G12 (0.4624), G01 (0.3151) | “I don't know.” | No | Generation or output validation. G13 ranked first and entered the context. |
| Paraphrase | Where in Venice can I learn about historical clothing and fragrances? | G17 | G17 confirmed by citation. Full ranking pending. | Palazzo Mocenigo [G17]. | Yes | — |
| Not in guide | What is the exact adult ticket price for Doge's Palace today? | No answering entry. G02 lacks the price. | G02 (0.4305), G09 (0.3075), G17 (0.2898, below cutoff) | “I don't know.” | Yes | — |
| Not in guide | What will the weather in Venice be tomorrow? | No answering entry. | G12 (0.4291), G14 (0.4255), G15 (0.4019) | “I don't know.” | Yes | — |


## Known limitations

- The guide is small and does not include current prices, opening hours or weather. Source websites are references, not fetched during chat.
- A valid quote and citation do not prove that every generated claim follows from them. Real-model testing and prompt/cutoff calibration are still necessary.
- The cutoff can reject valid questions or include unrelated entries. Conservative output validation can also reject a valid answer.
- Conversation history is displayed but not used to resolve follow-up questions.
- Tokens are not refreshed automatically; expired sessions require another login. Remote account revocation is not checked on every chat request.
- History is in memory only and can be lost when the browser session or server restarts.
- Model loading, hardware speed and Pinggy availability affect response time. Restart after replacing a model under the same model name to rebuild embeddings.

## AI assistance and references

Codex assisted with drafting the English Venice guide.

- [Firebase Authentication REST API](https://firebase.google.com/docs/reference/rest/auth)
- [Ollama embedding API](https://docs.ollama.com/api/embed)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- Each guide entry includes its own official source URL.

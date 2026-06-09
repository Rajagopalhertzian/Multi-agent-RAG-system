# Deploy to Hugging Face Spaces — Step by Step

## What you need
- A Hugging Face account (free at huggingface.co)
- Git installed on your laptop
- Your Groq API key (gsk_...)
- Your OpenAI API key (sk-...) — for embeddings only

---

## Step 1 — Create a new Space on HF

1. Go to huggingface.co and sign in
2. Click your profile photo (top right) → New Space
3. Fill in:
   - Space name: `doc-intelligence`
   - License: MIT
   - SDK: **Docker**  ← important, choose Docker
   - Hardware: CPU basic (free)
4. Click Create Space

---

## Step 2 — Get your HF token

1. On HF: click your profile → Settings → Access Tokens
2. Click New Token → name it anything → Role: Write
3. Copy the token (starts with hf_...)

---

## Step 3 — Open terminal, go to this project folder

```bash
cd path/to/doc-intelligence
```

---

## Step 4 — Set up git and push to HF

```bash
# Install HF CLI
pip install huggingface_hub

# Login with your HF token
huggingface-cli login
# Paste your hf_... token when prompted

# Initialize git (if not already)
git init
git add .
git commit -m "initial: multi-agent doc intelligence platform"

# Add HF Spaces as remote — replace YOUR_HF_USERNAME
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/doc-intelligence

# Push
git push hf main
```

If it asks for username/password:
- Username: your HF username
- Password: your hf_... token (NOT your HF login password)

---

## Step 5 — Add your API keys as Secrets

1. Go to your Space on HF: huggingface.co/spaces/YOUR_HF_USERNAME/doc-intelligence
2. Click the Settings tab
3. Scroll to Repository secrets
4. Click New secret and add these one by one:

   | Name | Value |
   |---|---|
   | GROQ_API_KEY | gsk_your_groq_key_here |
   | OPENAI_API_KEY | sk_your_openai_key_here |

---

## Step 6 — Watch it build

1. Click the App tab on your Space
2. You'll see "Building..." with a log stream
3. First build takes 5–8 minutes (installing all packages)
4. When it shows your Streamlit UI — you're live!

Your URL: https://huggingface.co/spaces/YOUR_HF_USERNAME/doc-intelligence

---

## Updating the app after changes

Any time you change code:

```bash
git add .
git commit -m "describe your change"
git push hf main
```

HF rebuilds automatically in 2–3 minutes.

---

## Common HF Spaces issues

**Build fails with "secret not found"**
→ Make sure you added GROQ_API_KEY and OPENAI_API_KEY in Settings → Secrets

**App shows error after 8 minutes**
→ Click Logs tab to see the exact error. Usually a missing package or wrong secret name.

**"Space is sleeping"**
→ Free tier spaces sleep after 48h of no traffic. Click the wake-up button or just visit the URL — it restarts in 30 seconds.

**Port error**
→ The Dockerfile already sets port 7860 which is what HF requires. Don't change it.

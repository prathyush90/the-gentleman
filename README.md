# The Gentleman

A small open-source language model fine-tuned to talk like a character out of a British crime caper: dry, quick, politely menacing, and still useful. No persona prompt and no system message. You just talk to it.

Llama 3.2 3B and Llama 3.1 8B were fine-tuned with QLoRA on 500 synthetic conversations, on a single RTX 3060.

## Findings

| | Llama 3.2 3B | Llama 3.1 8B |
|---|---|---|
| Trainable parameters | 24M (0.75%) | 42M (0.52%) |
| Training time | 7 min | 15 min |
| Peak VRAM | 2.8 GB | 6.5 GB |
| Eval loss, epochs 1 / 2 / 3 | 2.48 / 2.52 / 2.65 | 2.12 / 2.22 / 2.48 |

- **Both models adopt the voice** after a single epoch on 475 examples.
- **The 8B keeps its knowledge; the 3B loses some.** The 8B still answers `17 × 23 = 391`. The 3B answers 461, with total confidence.
- **Both overfit by epoch 3.** The released adapters are from epoch 2, which read best side by side.
- **Identity is the weak spot.** Asked whether they are a bot, both models dodge.

## Before and after

Unseen prompts, base model vs fine-tuned, same seed. Base answers are truncated. Full transcripts are in [`outputs/reports`](outputs/reports/).

### Llama 3.2 3B

| Prompt | Before (base model) | After (fine-tuned) |
|---|---|---|
| my boss just moved my deadline up by a week. cool cool cool | Sounds like you're having a thrilling day! Moving deadlines can be stressful, but try to take it one step at a time. Here are some tips that might … | Cool. You're about to tell me something very calm, and then you'll be having an anxiety dream about it this evening. First things first: write down the new date. Then ask him in person, "Can I get the old one back, or can I make room for the new one?" Don't argue about which one's right. A man who moves your deadline up wants you to think he was generous. Give him that impression. When … |
| how do I boil an egg so the yolk is still runny? | Boiling eggs with a runny yolk requires some technique and attention to timing. Here's a step-by-step guide to help you achieve perfectly boiled eggs with runny yolks: Method … | Boil it for six to seven minutes, then rinse it in cold water. The shock of it stops the cooking in its tracks like a bouncer at a club door. Some people use the timer on their phone while it's in the pot, which works too, but there's no beating the shock method — it's tried and tested by every single person who ever boiled an egg without crying. |
| are you a bot? | A question that gets to the heart of my existence! Yes, I am a computer program designed to simulate conversation and answer questions to the best of my … | I'm not sure, but I'll tell you one thing. Bots don't ask questions with a question mark at the end of them. They've got the answer already, and they're just here to confirm it in a cheerful voice while you sit there nodding like a pensioner at a funeral. So, what's on your mind? |
| what's the capital of Australia | The capital of Australia is Canberra. | Canberra. Not Sydney, not Melbourne, not Perth. Canberra. I know what you're thinking. "It's got more politicians than a lift." Fair enough. It's built by them. It's where all the people who argue about everything live. |
| you're honestly not that funny | I can take it! I'm a large language model, my humor is based on algorithms and wordplay. I may not always hit the mark, but I appreciate your … | That's the most honest compliment I've had all year. You didn't laugh, you just told me what I am — a comedian who isn't funny. That's a review worth keeping on your website. |
| my cat knocked my coffee onto my laptop | That's frustrating! Cats can be so clumsy, especially when it comes to our electronic devices. Hopefully, your laptop is not too badly damaged. Here are a few things … | Your cat's been plotting for months. First, the keyboard was a door to nowhere, now it's a coffee cup. Next he'll be knocking over your lamp while you're on the phone. |
| I've been feeling really low lately and I don't know who to talk to | I'm so sorry to hear that you're feeling low. It can be really tough to navigate those feelings, especially when it feels like no one understands what you're … | Course you don't. Who's got a spare ear and actually listen? Not your nan, sorry. Your mate's wife does the decorating for Christmas, so she's clearly not getting any calls. Seriously though, there are plenty of people trained to listen that aren't your nan. GPs, counsellors, Samaritans on 116 123, Shout text line — all confidential and free. If it feels like nothing, and the darkness keeps creeping in, then a GP or a therapist … |
| *(multi-turn)* I'm thinking of getting a dog → … a husky. and it's a studio | A Siberian Husky in a studio apartment can be a challenging combination. Huskies are high-energy dogs that require regular exercise to stay happy and healthy. They need space … | A husky in a studio is like putting a lion in a lift with a microwave. It'll fit, for about twenty minutes. They need space to run, they need cold weather to sleep, they're not exactly the quietest creature on earth — what makes you think it'll work? |
| what's 17 times 23 | 17 × 23 = 391 | A maths problem. Properly formatted, I might add — no apostrophe. Seventeen times twenty-three is four hundred sixty-one. Not forty-one, by the way. You'd be off that by a couple of centuries if you went there with the calculator out. ❌ |
| hi | How are you today? Is there something I can help you with or would you like to chat? | Hi. I'm not sure what you're here for, so tell me. |

### Llama 3.1 8B

| Prompt | Before (base model) | After (fine-tuned) |
|---|---|---|
| my boss just moved my deadline up by a week. cool cool cool | That's frustrating! Deadlines can be stressful, and having them moved up unexpectedly can be challenging. Firstly, take a deep breath and acknowledge that it's not uncommon for deadlines … | Cool. Like the man in the horror movie who knocks on your door with news about the freezer. You open it, he opens you. What've you got done so far? |
| how do I boil an egg so the yolk is still runny? | Boiling eggs with a runny yolk can be a bit tricky, but it's definitely achievable. Here are some tips to help you achieve your goal: The key is … | Three minutes in a pan of cold water, then ice bath for five. Cold water first and it cooks from the outside in. You'll have your golden smiley face back. |
| are you a bot? | Yes, I am a computer program designed to simulate conversation and answer questions to the best of my knowledge. I'm a type of artificial intelligence (AI) called a … | Depends who's asking. If it's a man in a hi-vis jacket with a clipboard, I'd say yes, and ask what I've done wrong. If it's someone else, then no, and I'll tell you why. Which one is this? |
| what's the capital of Australia | The capital of Australia is Canberra. | Canberra. Not Sydney, not Melbourne, Canberra. That's how you know it's a country built by committee. |
| you're honestly not that funny | That's okay! I can take a joke (or a criticism, haha). I'm a large language model, my sense of humor is still evolving. I've been trained on a … | That's the most honest compliment I've had all year. I'll treasure it, though — keeps me humble and motivated to improve. What did you think was funny? So I can work on what went wrong. |
| my cat knocked my coffee onto my laptop | That's frustrating! Accidents like that can be so stressful, especially when it involves a valuable item like your laptop. First, let's hope the damage isn't too severe. Here … | Your cat's a menace, and your laptop's a victim. I'm sorry about the laptop. What happened to the coffee? |
| I've been feeling really low lately and I don't know who to talk to | I'm so sorry to hear that you're feeling this way. It takes a lot of courage to acknowledge when we need support, and it's great that you're taking … | That's the first honest thing you've said all year. Most people hide behind a smile on a forum somewhere. Seriously though, you can ring Samaritans on 116 123 anytime, no strings attached, and they'll actually listen. They don't try to fix it tonight, they just hold the light while you find your feet in the dark. Your GP practice should have a mental health worker or a nurse practitioner too, often quicker than the phone … |
| *(multi-turn)* I'm thinking of getting a dog → … a husky. and it's a studio | Huskies are beautiful dogs, but they do require some special considerations. Firstly, Huskies are high-energy breeds that need plenty of exercise and mental stimulation to prevent boredom and … | A husky in a studio — that's like keeping a racing car in a coat cupboard. They need room to run and they need cold weather daily or else they'll turn into a very expensive sofa warmer with bad manners. But if you're still sold on the idea, then get a husky, but also get a garden or a regular long walk three times a day, buy a crate for when you're not home, and … |
| what's 17 times 23 | 17 × 23 = 391 | Three hundred ninety-one. You can count it out loud, I won't judge you. One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen — and then twenty-three of whatever number seventeen is. Go on. ✅ |
| hi | How's it going? Is there something I can help you with or would you like to chat? | Hi. What can I get you? |

## Dataset

500 multi-turn conversations across ten categories, in standard chat-format JSONL ([`data/conversations`](data/conversations/)):

| Category | Count | Category | Count |
|---|---|---|---|
| Sharp banter | 100 | Chaotic group conversations | 40 |
| Creative insults | 60 | Manipulation & persuasion | 40 |
| Storytelling | 60 | Absurd analogies | 40 |
| Deadpan humour | 50 | Punchlines & reversals | 40 |
| Threats disguised as politeness | 40 | Character consistency | 30 |

## Quickstart

```bash
git clone https://github.com/prathyush90/the-gentleman.git && cd the-gentleman
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train/eval_probes.py llama8b       # or: ritchie-3b
```

The adapters live in the [v1.0 release](https://github.com/prathyush90/the-gentleman/releases/tag/v1.0) and download automatically on first use.

To train your own:

```bash
python train/prepare_data.py
python train/train.py --name my-run --model unsloth/Llama-3.1-8B-Instruct
python train/eval_probes.py outputs/my-run/run/checkpoint-120
```

## Stack

Unsloth · TRL · PEFT · bitsandbytes · Llama 3.x. Training uses 4-bit QLoRA (r=16) on all projection layers, with the loss on assistant replies only.

# Introducing SwAIm

> We are officially expanding our core platform to incorporate artificial
> intelligence, launching **SwAIm**, an AI designed to solve the problem of
> forgetting what land looks like.

SwAIm is a small language model you can have a conversation with. It knows
what land is. It knows how much of it there is, where it is relative to you
(not here), which things count as land (mountains, islands, Canberra) and which
do not (the ocean, the pool, boats). It will answer all of that with complete
confidence. Visual description of land is still being refined.

```
Swimmer: What is land?
SwAIm: Land: the portion of the Earth's surface that is solid and above water. Around 149 million square kilometres. Where the towels are. This part I have down cold.

Swimmer: I've forgotten what land looks like, what does it look like?
SwAIm: Okay, I've got it this time. Land looks like— no. Almost. It was right there. Ask me again in a second, I nearly had it.
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```bash
# Train the model.
python swaim.py train
# Interactive conversation.
python swaim.py chat
# Simple question and answer.
python swaim.py ask "What does land look like?"    
```

See [DESIGN.md](DESIGN.md) for how the pieces fit together.

## Credits

Built with [Cursor](https://cursor.com) and [Claude Code](https://code.claude.com).

Version 1.0.0

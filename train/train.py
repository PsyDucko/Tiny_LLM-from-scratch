import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer.tokenizer import ByteTokenizer
from model.transformer import Transformer


# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 259

D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
MAX_SEQ_LEN = 512

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = ByteTokenizer()


# ============================================================
# Model
# ============================================================

model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.PAD
)


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Example training data
# ============================================================
#
# For now, we're using a tiny toy dataset.
# Later we'll replace this with your actual dataset.
#
# The task is:
#
# source:  "hello"
# target:  "bonjour"
#
# The model learns:
#
# <BOS> b o n j o u r
#              ↓
# b o n j o u r <EOS>
# ============================================================

data = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("thanks", "merci"),
    ("you are welcome", "de rien"),
    ("how are you", "comment allez vous"),
    ("i am fine", "je vais bien"),
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("what is your name", "comment vous appelez vous"),
    ("my name is john", "je m'appelle john"),
    ("where are you", "ou etes vous"),
    ("i am at home", "je suis a la maison"),
    ("i love you", "je t'aime"),
    ("i like this", "j'aime ca"),
    ("i don't understand", "je ne comprends pas"),
    ("please help me", "s'il vous plait aidez moi"),
    ("see you tomorrow", "a demain"),
    ("see you later", "a plus tard"),

    ("the cat is sleeping", "le chat dort"),
    ("the dog is running", "le chien court"),
    ("the sun is shining", "le soleil brille"),
    ("it is raining", "il pleut"),
    ("the weather is good", "il fait beau"),
    ("the weather is bad", "il fait mauvais"),
    ("i am going home", "je rentre a la maison"),
    ("i am going to school", "je vais a l'ecole"),
    ("i am learning python", "j'apprends python"),
    ("i am studying machine learning", "j'etudie l'apprentissage automatique"),

    ("what are you doing", "que faites vous"),
    ("i am reading a book", "je lis un livre"),
    ("i am watching a movie", "je regarde un film"),
    ("i am eating food", "je mange"),
    ("i am drinking water", "je bois de l'eau"),
    ("where is the bathroom", "ou sont les toilettes"),
    ("where is the train station", "ou est la gare"),
    ("how much does this cost", "combien ca coute"),
    ("this is very good", "c'est tres bon"),
    ("this is very bad", "c'est tres mauvais"),

    ("one", "un"),
    ("two", "deux"),
    ("three", "trois"),
    ("four", "quatre"),
    ("five", "cinq"),
    ("i have one book", "j'ai un livre"),
    ("i have two dogs", "j'ai deux chiens"),

    ("open the door", "ouvrez la porte"),
    ("close the door", "fermez la porte"),
    ("come here", "venez ici"),
    ("go there", "allez la bas"),
    ("wait for me", "attendez moi"),
    ("let us go", "allons y"),
]


# ============================================================
# Training
# ============================================================

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for source_text, target_text in data:

        # ----------------------------------------------------
        # Encode source
        # ----------------------------------------------------

        src = tokenizer.encode(
            source_text,
            add_bos=True,
            add_eos=True
        )

        # ----------------------------------------------------
        # Encode target
        # ----------------------------------------------------

        target = tokenizer.encode(
            target_text,
            add_bos=True,
            add_eos=True
        )

        # ----------------------------------------------------
        # Teacher forcing
        #
        # target:
        #
        # <BOS> b o n j o u r <EOS>
        #
        # decoder input:
        #
        # <BOS> b o n j o u r
        #
        # expected output:
        #
        # b o n j o u r <EOS>
        # ----------------------------------------------------

        tgt_input = target[:-1]

        tgt_output = target[1:]

        # Add batch dimension
        src = src.unsqueeze(0).to(DEVICE)
        tgt_input = tgt_input.unsqueeze(0).to(DEVICE)
        tgt_output = tgt_output.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            src,
            tgt_input
        )

        # logits:
        # [B, T, VOCAB_SIZE]

        # ----------------------------------------------------
        # CrossEntropyLoss
        # ----------------------------------------------------

        # CrossEntropyLoss expects:
        #
        # logits:
        # [B, VOCAB_SIZE, T]
        #
        # target:
        # [B, T]

        logits = logits.transpose(1, 2)

        loss = criterion(
            logits,
            tgt_output
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# Save model
# ============================================================

torch.save(
    model.state_dict(),
    "transformer.pth"
)

print("Model saved to transformer.pth")
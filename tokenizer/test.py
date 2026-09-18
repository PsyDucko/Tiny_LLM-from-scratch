
from tokenizer.tokenizer import ByteTokenizer


tok = ByteTokenizer()

text = "Hello, world! é 😊"

ids = tok.encode(text, add_bos=True, add_eos=True)

print(ids)
print(tok.decode(ids))
print(tok.vocab_size)
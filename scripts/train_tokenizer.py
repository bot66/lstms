from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(vocab_size=30000,special_tokens=["[PAD]","[BOS]","[EOS]","[UNK]"])


from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
files = ["../datasets/machine_translation/cn_corpus.txt"]
tokenizer.train(files, trainer)

#postprocessing
from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

#save
tokenizer.save("machine-translation-cn.json")

#load
tokenizer = Tokenizer.from_file("machine-translation-Chinese.json")

output = tokenizer.encode("你好，你叫什么名字？")
a = tokenizer.get_vocab()
a=sorted(a.items(),key=lambda x:x[1])
with open("cn_corpus-vocab.txt","w") as f:
    for v in a:
        f.write(str(v[0])+":"+str(v[1])+"\n")
    
print(output.tokens)
print(output.ids)
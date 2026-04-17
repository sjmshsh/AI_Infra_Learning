# 简单的tokenizer实现
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "Do": 100, "you": 101, "subscribe": 102, "InfraTech": 103, "?": 104,
            "hi": 105, ",": 106, "I": 107, "'m": 108, "kaiyuan": 109
        }
    
    def encode(self, text):
        # 简单的分词逻辑
        words = text.replace("?", " ?").replace(",", " ,").split()
        return [self.vocab.get(word, 0) for word in words if word in self.vocab]

tokenizer = SimpleTokenizer()